from __future__ import annotations

import os
import re
import time
from difflib import unified_diff
from contextlib import contextmanager
from dataclasses import replace
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
    Constraint,
    DimExprBinary,
    TypeBool,
    TypeAny,
    TypeDim,
    TypeExpr,
    TypeList,
    TypeNull,
    TypeOptional,
    TypePath,
    TypeTensor,
    TypeTuple,
    ast_equal,
)
from ..ast.render import render_axon_file
from ..entrypoint import resolve_main_module
from ..resolve import prune_unreachable_definitions
from ..typecheck2 import typecheck2_flat_axon_file
from ..validate import (
    validate_backend_required_flat_typed_axon_file,
    validate_flat_axon_file,
    validate_typed_axon_file,
)
from ..validate.optimized import validate_optimized_flat_typed_axon_file


_OPT_DEBUG_ENV = "AXON_OPTIMIZE_DEBUG"
_OPT_DEBUG_ONE_PASS_ENV = "AXON_OPTIMIZE_DEBUG_ONE_PASS"
_OPT_DEBUG_DIFF_ENV = "AXON_OPTIMIZE_DEBUG_DIFF"
_OPT_DEBUG_ACTIVE = False
_OPT_DEBUG_PASS = 0
_OPT_DEBUG_STATS: dict[str, float | int] = {}


def _opt_debug_enabled(*, pass_index: int) -> bool:
    if os.environ.get(_OPT_DEBUG_ENV) not in {"1", "true", "TRUE", "yes", "YES"}:
        return False
    if os.environ.get(_OPT_DEBUG_ONE_PASS_ENV) in {"1", "true", "TRUE", "yes", "YES"}:
        return pass_index == 1
    return True


def _opt_debug_begin_pass(*, pass_index: int) -> None:
    global _OPT_DEBUG_ACTIVE, _OPT_DEBUG_PASS, _OPT_DEBUG_STATS
    _OPT_DEBUG_ACTIVE = _opt_debug_enabled(pass_index=pass_index)
    _OPT_DEBUG_PASS = pass_index
    _OPT_DEBUG_STATS = {}


def _opt_debug_record(counter: str, seconds_key: str, elapsed: float) -> None:
    if not _OPT_DEBUG_ACTIVE:
        return
    _OPT_DEBUG_STATS[counter] = int(_OPT_DEBUG_STATS.get(counter, 0)) + 1
    _OPT_DEBUG_STATS[seconds_key] = float(_OPT_DEBUG_STATS.get(seconds_key, 0.0)) + elapsed


@contextmanager
def _opt_debug_time(name: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        _opt_debug_record(f"{name}_calls", f"{name}_seconds", time.perf_counter() - start)


def _opt_debug_end_pass(*, modules: int) -> None:
    if not _OPT_DEBUG_ACTIVE:
        return
    keys = sorted(key[:-8] for key in _OPT_DEBUG_STATS if key.endswith("_seconds"))
    print(f"[axon-optimize] pass={_OPT_DEBUG_PASS} modules={modules}")
    for key in keys:
        print(
            "[axon-optimize]"
            f" pass={_OPT_DEBUG_PASS}"
            f" fn={key}"
            f" calls={int(_OPT_DEBUG_STATS.get(f'{key}_calls', 0))}"
            f" seconds={float(_OPT_DEBUG_STATS.get(f'{key}_seconds', 0.0)):.6f}"
        )


def _opt_debug_print_diff(*, before: AxonFile, after: AxonFile, pass_index: int) -> None:
    if os.environ.get(_OPT_DEBUG_DIFF_ENV) not in {"1", "true", "TRUE", "yes", "YES"}:
        return
    before_lines = render_axon_file(before, show_types=True).splitlines()
    after_lines = render_axon_file(after, show_types=True).splitlines()
    diff_lines = list(
        unified_diff(
            before_lines,
            after_lines,
            fromfile=f"pass_{pass_index}_input.axon",
            tofile=f"pass_{pass_index}_output.axon",
            lineterm="",
        )
    )
    print(f"[axon-optimize-diff] pass={pass_index} lines={len(diff_lines)}")
    for line in diff_lines:
        print(line)


def _is_atomic_expr(expr: AxonExpr) -> bool:
    if isinstance(expr, AxonExprAscribe):
        return _is_atomic_expr(expr.expr)
    if isinstance(expr, AxonExprParen):
        return _is_atomic_expr(expr.inner)
    if isinstance(
        expr,
        (
            AxonExprName,
            AxonExprInt,
            AxonExprFloat,
            AxonExprBool,
            AxonExprNull,
            AxonExprString,
            AxonExprPath,
        ),
    ):
        return True
    if isinstance(expr, AxonExprList | AxonExprTuple):
        return all(_is_atomic_expr(item) for item in expr.items)
    return False


def _unwrap_expr(expr: AxonExpr) -> AxonExpr:
    current = expr
    while isinstance(current, AxonExprAscribe | AxonExprParen):
        if isinstance(current, AxonExprAscribe):
            current = current.expr
            continue
        current = current.inner
    return current


def _expr_names(expr: AxonExpr) -> set[str]:
    if isinstance(expr, AxonExprName):
        return {expr.name}
    if isinstance(expr, AxonExprBinary):
        return _expr_names(expr.left) | _expr_names(expr.right)
    if isinstance(expr, AxonExprBind):
        return _expr_names(expr.value) | (_expr_names(expr.body) - {expr.var})
    if isinstance(expr, AxonExprCall):
        names: set[str] = set()
        for arg in expr.args:
            names.update(_expr_names(arg))
        for value in expr.kwargs.values():
            if isinstance(value, AxonExpr):
                names.update(_expr_names(value))
        return names
    if isinstance(expr, AxonExprDo):
        return _stmt_names(expr.body)
    if isinstance(expr, AxonExprIf | AxonExprTernary):
        return _expr_names(expr.cond) | _expr_names(expr.true_expr) | _expr_names(expr.false_expr)
    if isinstance(expr, AxonExprLambda):
        return _expr_names(expr.body) - {expr.var}
    if isinstance(expr, AxonExprAscribe):
        return _expr_names(expr.expr)
    if isinstance(expr, AxonExprList | AxonExprTuple):
        item_names: set[str] = set()
        for item in expr.items:
            item_names.update(_expr_names(item))
        return item_names
    if isinstance(expr, AxonExprParen):
        return _expr_names(expr.inner)
    if isinstance(expr, AxonExprPipe):
        names = _expr_names(expr.value)
        for item in expr.stages:
            names.update(_expr_names(item))
        return names
    return set()


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_GENERATED_HELPER_RE = re.compile(
    r"^(?P<base>.+?)(?:__cond_(?:true|else)_\d+|__loop_[A-Za-z0-9_]+_\d+)$"
)


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


def _type_expr_dim_names(tp: TypeExpr | None) -> set[str]:
    return _type_dim_names(tp)


def _inferred_dim_names_expr(expr: AxonExpr) -> set[str]:
    names = _type_expr_dim_names(expr.inferred_type)
    if expr.inferred_dims is not None:
        names.update(_constraint_operand_names(expr.inferred_dims))
    if isinstance(expr, AxonExprBinary):
        names.update(_inferred_dim_names_expr(expr.left))
        names.update(_inferred_dim_names_expr(expr.right))
    elif isinstance(expr, AxonExprBind):
        names.update(_inferred_dim_names_expr(expr.value))
        names.update(_inferred_dim_names_expr(expr.body))
    elif isinstance(expr, AxonExprCall):
        for arg in expr.args:
            names.update(_inferred_dim_names_expr(arg))
        for value in expr.kwargs.values():
            if isinstance(value, AxonExpr):
                names.update(_inferred_dim_names_expr(value))
    elif isinstance(expr, AxonExprDo):
        names.update(_inferred_dim_names_stmts(expr.body))
    elif isinstance(expr, AxonExprIf | AxonExprTernary):
        names.update(_inferred_dim_names_expr(expr.cond))
        names.update(_inferred_dim_names_expr(expr.true_expr))
        names.update(_inferred_dim_names_expr(expr.false_expr))
    elif isinstance(expr, AxonExprLambda):
        names.update(_inferred_dim_names_expr(expr.body))
    elif isinstance(expr, AxonExprAscribe):
        names.update(_inferred_dim_names_expr(expr.expr))
    elif isinstance(expr, AxonExprList | AxonExprTuple):
        for item in expr.items:
            names.update(_inferred_dim_names_expr(item))
    elif isinstance(expr, AxonExprParen):
        names.update(_inferred_dim_names_expr(expr.inner))
    elif isinstance(expr, AxonExprPipe):
        names.update(_inferred_dim_names_expr(expr.value))
        for item in expr.stages:
            names.update(_inferred_dim_names_expr(item))
    return {name for name in names if isinstance(name, str) and name.isidentifier()}


def _inferred_dim_names_stmts(statements: tuple[AxonStatement, ...]) -> set[str]:
    names: set[str] = set()
    for stmt in statements:
        if isinstance(stmt, AxonBind):
            names.update(_inferred_dim_names_expr(stmt.expr))
        elif isinstance(stmt, AxonReturn | AxonYield):
            for value in stmt.values:
                names.update(_inferred_dim_names_expr(value))
        elif isinstance(stmt, AxonCond):
            names.update(_inferred_dim_names_expr(stmt.cond))
            names.update(_inferred_dim_names_stmts(stmt.true_body))
            names.update(_inferred_dim_names_stmts(stmt.false_body))
        elif isinstance(stmt, AxonRepeat):
            names.update(_inferred_dim_names_expr(stmt.from_expr))
            names.update(_inferred_dim_names_expr(stmt.to_expr))
            names.update(_inferred_dim_names_expr(stmt.step_expr))
            names.update(_inferred_dim_names_stmts(stmt.body))
        elif isinstance(stmt, AxonScopeBind):
            names.update(_inferred_dim_names_expr(stmt.prefix))
            for raw_value in stmt.kwargs.values():
                if isinstance(raw_value, AxonExpr):
                    names.update(_inferred_dim_names_expr(raw_value))
            names.update(_inferred_dim_names_stmts(stmt.body))
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


def _stmt_expr_names(statements: tuple[AxonStatement, ...]) -> set[str]:
    names: set[str] = set()
    for stmt in statements:
        if isinstance(stmt, AxonBind):
            names.update(_expr_names(stmt.expr))
        elif isinstance(stmt, AxonReturn | AxonYield):
            for value in stmt.values:
                names.update(_expr_names(value))
        elif isinstance(stmt, AxonCond):
            names.update(_expr_names(stmt.cond))
            names.update(_stmt_expr_names(stmt.true_body))
            names.update(_stmt_expr_names(stmt.false_body))
        elif isinstance(stmt, AxonRepeat):
            names.update(_expr_names(stmt.from_expr))
            names.update(_expr_names(stmt.to_expr))
            names.update(_expr_names(stmt.step_expr))
            names.update(_stmt_expr_names(stmt.body))
        elif isinstance(stmt, AxonScopeBind):
            names.update(_expr_names(stmt.prefix))
            for raw_value in stmt.kwargs.values():
                if isinstance(raw_value, AxonExpr):
                    names.update(_expr_names(raw_value))
            names.update(_stmt_expr_names(stmt.body))
    return names


def _inline_free_names(statements: tuple[AxonStatement, ...]) -> set[str]:
    return _stmt_expr_names(statements) - _bound_names_statements(statements)


def _module_referenced_dim_names_after_subst(
    module: AxonDefinition, subst: Mapping[str, AxonExpr]
) -> set[str]:
    rewritten = _substitute_stmts(module.statements, subst)
    names = _stmt_param_like_names(rewritten)
    names.update(_inferred_dim_names_stmts(rewritten))
    names.update(_type_dim_names(module.return_type_expr))
    for constraint in module.constraints or ():
        if not _is_trivial_identity_constraint(constraint):
            names.update(_constraint_names(constraint))
    return {name for name in names if isinstance(name, str) and name.isidentifier()}


def _filter_dim_safe_param_subst(module: AxonDefinition, subst: Mapping[str, AxonExpr]) -> dict[str, AxonExpr]:
    filtered = dict(subst)
    params_by_name = {param.name: param for param in module.params}
    changed = True
    while changed:
        changed = False
        referenced_dims = _module_referenced_dim_names_after_subst(module, filtered)
        for name in tuple(filtered):
            param = params_by_name.get(name)
            if param is None:
                continue
            param_dims = _type_dim_names(param.type_expr)
            if param_dims & referenced_dims:
                filtered.pop(name)
                changed = True
    return filtered


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


def _count_name_uses_expr(expr: AxonExpr, counts: dict[str, int]) -> None:
    if isinstance(expr, AxonExprName):
        counts[expr.name] = counts.get(expr.name, 0) + 1
        return
    if isinstance(expr, AxonExprBinary):
        _count_name_uses_expr(expr.left, counts)
        _count_name_uses_expr(expr.right, counts)
        return
    if isinstance(expr, AxonExprBind):
        _count_name_uses_expr(expr.value, counts)
        _count_name_uses_expr(expr.body, counts)
        return
    if isinstance(expr, AxonExprCall):
        for arg in expr.args:
            _count_name_uses_expr(arg, counts)
        for value in expr.kwargs.values():
            if isinstance(value, AxonExpr):
                _count_name_uses_expr(value, counts)
        return
    if isinstance(expr, AxonExprDo):
        _count_name_uses_stmts(expr.body, counts)
        return
    if isinstance(expr, AxonExprIf | AxonExprTernary):
        _count_name_uses_expr(expr.cond, counts)
        _count_name_uses_expr(expr.true_expr, counts)
        _count_name_uses_expr(expr.false_expr, counts)
        return
    if isinstance(expr, AxonExprLambda):
        _count_name_uses_expr(expr.body, counts)
        return
    if isinstance(expr, AxonExprAscribe):
        _count_name_uses_expr(expr.expr, counts)
        return
    if isinstance(expr, AxonExprList | AxonExprTuple):
        for item in expr.items:
            _count_name_uses_expr(item, counts)
        return
    if isinstance(expr, AxonExprParen):
        _count_name_uses_expr(expr.inner, counts)


def _count_name_uses_stmts(statements: tuple[AxonStatement, ...], counts: dict[str, int]) -> None:
    for stmt in statements:
        if isinstance(stmt, AxonBind):
            _count_name_uses_expr(stmt.expr, counts)
        elif isinstance(stmt, AxonReturn | AxonYield):
            for value in stmt.values:
                _count_name_uses_expr(value, counts)
        elif isinstance(stmt, AxonCond):
            _count_name_uses_expr(stmt.cond, counts)
            _count_name_uses_stmts(stmt.true_body, counts)
            _count_name_uses_stmts(stmt.false_body, counts)
        elif isinstance(stmt, AxonRepeat):
            _count_name_uses_expr(stmt.from_expr, counts)
            _count_name_uses_expr(stmt.to_expr, counts)
            _count_name_uses_expr(stmt.step_expr, counts)
            _count_name_uses_stmts(stmt.body, counts)
        elif isinstance(stmt, AxonScopeBind):
            for raw_value in stmt.kwargs.values():
                if isinstance(raw_value, AxonExpr):
                    _count_name_uses_expr(raw_value, counts)
            _count_name_uses_stmts(stmt.body, counts)


def _return_position_names(statements: tuple[AxonStatement, ...]) -> set[str]:
    names: set[str] = set()
    for stmt in statements:
        if isinstance(stmt, AxonReturn | AxonYield):
            for value in stmt.values:
                names.update(_expr_names(value))
        elif isinstance(stmt, AxonCond):
            names.update(_return_position_names(stmt.true_body))
            names.update(_return_position_names(stmt.false_body))
        elif isinstance(stmt, AxonRepeat):
            names.update(_return_position_names(stmt.body))
        elif isinstance(stmt, AxonScopeBind):
            names.update(_return_position_names(stmt.body))
    return names


def _folded_expr_eq(left: AxonExpr, right: AxonExpr) -> bool:
    return ast_equal(left, right)


def _substitute_path(expr: AxonExprPath, subst: Mapping[str, AxonExpr]) -> AxonExprPath:
    if len(expr.parts) == 1:
        direct = subst.get(expr.parts[0])
        if isinstance(direct, AxonExprPath):
            return direct
    rewritten_parts: list[str] = []
    for part in expr.parts:
        if part.startswith("{") and part.endswith("}") and len(part) > 2:
            inner = part[1:-1]
            replacement = subst.get(inner)
            if isinstance(replacement, AxonExprPath):
                rewritten_parts.extend(replacement.parts)
                continue
        rewritten_parts.append(part)
    return replace(expr, parts=tuple(rewritten_parts))


def _rename_path(expr: AxonExprPath, renames: Mapping[str, str]) -> AxonExprPath:
    rewritten_parts: list[str] = []
    for part in expr.parts:
        if part.startswith("{") and part.endswith("}") and len(part) > 2:
            inner = part[1:-1]
            rewritten_parts.append("{" + renames.get(inner, inner) + "}")
            continue
        rewritten_parts.append(part)
    return replace(expr, parts=tuple(rewritten_parts))


def _substitute_expr(
    expr: AxonExpr, subst: Mapping[str, AxonExpr], *, for_return: bool = False
) -> AxonExpr:
    with _opt_debug_time("substitute_expr"):
        if isinstance(expr, AxonExprName) and expr.name in subst:
            replacement = subst[expr.name]
            if for_return and not isinstance(replacement, AxonExprName):
                return expr
            return replacement
        if isinstance(expr, AxonExprPath):
            return _substitute_path(expr, subst)
        if isinstance(expr, AxonExprBinary):
            return replace(
                expr,
                left=_substitute_expr(expr.left, subst),
                right=_substitute_expr(expr.right, subst),
            )
        if isinstance(expr, AxonExprBind):
            next_subst = dict(subst)
            next_subst.pop(expr.var, None)
            return replace(
                expr,
                value=_substitute_expr(expr.value, subst),
                body=_substitute_expr(expr.body, next_subst),
            )
        if isinstance(expr, AxonExprCall):
            return replace(
                expr,
                args=tuple(_substitute_expr(arg, subst) for arg in expr.args),
                kwargs={
                    key: _substitute_expr(value, subst) if isinstance(value, AxonExpr) else value
                    for key, value in expr.kwargs.items()
                },
            )
        if isinstance(expr, AxonExprDo):
            return replace(expr, body=_substitute_stmts(expr.body, subst))
        if isinstance(expr, AxonExprIf | AxonExprTernary):
            return replace(
                expr,
                cond=_substitute_expr(expr.cond, subst),
                true_expr=_substitute_expr(expr.true_expr, subst),
                false_expr=_substitute_expr(expr.false_expr, subst),
            )
        if isinstance(expr, AxonExprLambda):
            next_subst = dict(subst)
            next_subst.pop(expr.var, None)
            return replace(expr, body=_substitute_expr(expr.body, next_subst))
        if isinstance(expr, AxonExprAscribe):
            return replace(expr, expr=_substitute_expr(expr.expr, subst, for_return=for_return))
        if isinstance(expr, AxonExprList | AxonExprTuple):
            return replace(expr, items=tuple(_substitute_expr(item, subst) for item in expr.items))
        if isinstance(expr, AxonExprParen):
            return replace(expr, inner=_substitute_expr(expr.inner, subst))
        if isinstance(expr, AxonExprPipe):
            return replace(
                expr,
                value=_substitute_expr(expr.value, subst),
                stages=tuple(_substitute_expr(item, subst) for item in expr.stages),
            )
        return expr


def _rename_expr(expr: AxonExpr, renames: Mapping[str, str]) -> AxonExpr:
    if isinstance(expr, AxonExprName) and expr.name in renames:
        return replace(expr, name=renames[expr.name])
    if isinstance(expr, AxonExprPath):
        return _rename_path(expr, renames)
    if isinstance(expr, AxonExprBinary):
        return replace(
            expr,
            left=_rename_expr(expr.left, renames),
            right=_rename_expr(expr.right, renames),
        )
    if isinstance(expr, AxonExprBind):
        next_renames = dict(renames)
        next_renames.pop(expr.var, None)
        return replace(
            expr,
            value=_rename_expr(expr.value, renames),
            body=_rename_expr(expr.body, next_renames),
        )
    if isinstance(expr, AxonExprCall):
        return replace(
            expr,
            args=tuple(_rename_expr(arg, renames) for arg in expr.args),
            kwargs={
                key: _rename_expr(value, renames) if isinstance(value, AxonExpr) else value
                for key, value in expr.kwargs.items()
            },
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
    current_renames = dict(renames)
    for stmt in statements:
        if isinstance(stmt, AxonBind):
            rewritten_targets = tuple(
                current_renames.get(target, target) for target in stmt.targets
            )
            rewritten.append(
                replace(
                    stmt,
                    targets=rewritten_targets,
                    expr=_rename_expr(stmt.expr, current_renames),
                )
            )
        elif isinstance(stmt, AxonReturn | AxonYield):
            rewritten.append(
                replace(
                    stmt,
                    values=tuple(_rename_expr(value, current_renames) for value in stmt.values),
                )
            )
        elif isinstance(stmt, AxonCond):
            rewritten.append(
                replace(
                    stmt,
                    cond=_rename_expr(stmt.cond, current_renames),
                    true_body=_rename_stmts(stmt.true_body, dict(current_renames)),
                    false_body=_rename_stmts(stmt.false_body, dict(current_renames)),
                )
            )
        elif isinstance(stmt, AxonRepeat):
            next_renames = dict(current_renames)
            if stmt.var in next_renames:
                next_renames.pop(stmt.var, None)
            for target in stmt.targets or ():
                next_renames.pop(target, None)
            for carry in stmt.carry or ():
                next_renames.pop(carry, None)
            rewritten.append(
                replace(
                    stmt,
                    var=current_renames.get(stmt.var, stmt.var),
                    targets=tuple(current_renames.get(name, name) for name in stmt.targets or ())
                    or stmt.targets,
                    carry=tuple(current_renames.get(name, name) for name in stmt.carry or ())
                    or stmt.carry,
                    from_expr=_rename_expr(stmt.from_expr, current_renames),
                    to_expr=_rename_expr(stmt.to_expr, current_renames),
                    step_expr=_rename_expr(stmt.step_expr, current_renames),
                    body=_rename_stmts(stmt.body, next_renames),
                )
            )
        elif isinstance(stmt, AxonScopeBind):
            rewritten_targets = tuple(
                current_renames.get(target, target) for target in stmt.targets
            )
            rewritten.append(
                replace(
                    stmt,
                    targets=rewritten_targets,
                    prefix=_rename_path(stmt.prefix, current_renames),
                    body=_rename_stmts(stmt.body, dict(current_renames)),
                    kwargs={
                        key: _rename_expr(value, current_renames)
                        if isinstance(value, AxonExpr)
                        else value
                        for key, value in stmt.kwargs.items()
                    },
                )
            )
    return tuple(rewritten)


def _bound_names_statements(statements: tuple[AxonStatement, ...]) -> set[str]:
    names: set[str] = set()
    for stmt in statements:
        if isinstance(stmt, AxonBind):
            names.update(target for target in stmt.targets if target != "_")
        elif isinstance(stmt, AxonCond):
            names.update(_bound_names_statements(stmt.true_body))
            names.update(_bound_names_statements(stmt.false_body))
        elif isinstance(stmt, AxonRepeat):
            names.add(stmt.var)
            names.update(target for target in stmt.targets or () if target != "_")
            names.update(carry for carry in stmt.carry or () if carry != "_")
            names.update(_bound_names_statements(stmt.body))
        elif isinstance(stmt, AxonScopeBind):
            names.update(target for target in stmt.targets if target != "_")
            names.update(_bound_names_statements(stmt.body))
    return names


def _fresh_name(base: str, used: set[str]) -> str:
    stem = f"__inl_{base}"
    if stem not in used:
        used.add(stem)
        return stem
    idx = 1
    while True:
        candidate = f"{stem}_{idx}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        idx += 1


def _freshen_inline_module_statements(
    module: AxonDefinition,
    *,
    blocked_names: set[str],
) -> tuple[tuple[AxonStatement, ...], dict[str, str]]:
    renames: dict[str, str] = {}
    used = set(blocked_names)
    used.update(_param_names(module))
    for name in sorted(_bound_names_statements(module.statements)):
        if name in blocked_names:
            renames[name] = _fresh_name(name, used)
        else:
            used.add(name)
    if not renames:
        return module.statements, {}
    return _rename_stmts(module.statements, renames), renames


def _replace_returns_with_bind(
    statements: tuple[AxonStatement, ...],
    *,
    targets: tuple[str, ...],
    call_expr: AxonExpr,
) -> tuple[AxonStatement, ...]:
    rewritten: list[AxonStatement] = []
    for stmt in statements:
        if isinstance(stmt, AxonReturn):
            if len(targets) == 1 and len(stmt.values) == 1:
                rewritten.append(AxonBind(targets=targets, expr=stmt.values[0]))
                continue
            if len(targets) > 1 and len(stmt.values) == 1:
                rewritten.append(AxonBind(targets=targets, expr=stmt.values[0]))
                continue
            tuple_expr = AxonExprTuple(
                items=stmt.values,
                inferred_type=call_expr.inferred_type,
                inferred_arity=call_expr.inferred_arity,
                inferred_dims=call_expr.inferred_dims,
            )
            rewritten.append(AxonBind(targets=targets, expr=tuple_expr))
            continue
        rewritten.append(stmt)
    return tuple(rewritten)


def _expr_to_dim_token_inline(expr: AxonExpr) -> int | str | DimExprBinary | None:
    inner = _unwrap_expr(expr)
    if isinstance(inner, AxonExprInt):
        return inner.value
    if isinstance(inner, AxonExprName):
        return inner.name
    if isinstance(inner, AxonExprBinary) and inner.op in {"+", "-", "*", "/"}:
        left = _expr_to_dim_token_inline(inner.left)
        right = _expr_to_dim_token_inline(inner.right)
        if left is None or right is None:
            return None
        return DimExprBinary(op=inner.op, left=left, right=right)
    return None


def _type_dim_names(tp: TypeExpr | None) -> set[str]:
    if tp is None:
        return set()
    if isinstance(tp, TypeOptional):
        return _type_dim_names(tp.inner)
    if isinstance(tp, TypeTensor):
        names: set[str] = set()
        for dim in tp.dims:
            names.update(
                name
                for name in _constraint_operand_names(dim)
                if isinstance(name, str) and name.isidentifier()
            )
        return names
    if isinstance(tp, TypeTuple):
        tuple_names: set[str] = set()
        for item in tp.items:
            tuple_names.update(_type_dim_names(item))
        return tuple_names
    if isinstance(tp, TypeList):
        return _type_dim_names(tp.item)
    return set()


def _module_dim_names(module: AxonDefinition) -> set[str]:
    names: set[str] = set()
    for param in module.params:
        names.update(_type_dim_names(param.type_expr))
    names.update(_type_dim_names(module.return_type_expr))
    return names


def _bind_inline_dim_tokens(
    expected: tuple[int | str | DimExprBinary, ...],
    actual: tuple[int | str | DimExprBinary, ...],
    *,
    subst: dict[str, int | str | DimExprBinary | tuple[int | str | DimExprBinary, ...]],
) -> None:
    variadic_positions = [
        idx for idx, dim in enumerate(expected) if isinstance(dim, str) and dim.startswith("..")
    ]
    if not variadic_positions:
        if len(expected) != len(actual):
            return
        for exp_dim, act_dim in zip(expected, actual, strict=True):
            if isinstance(exp_dim, str):
                subst.setdefault(exp_dim, act_dim)
        return
    if len(variadic_positions) != 1:
        return
    variadic_idx = variadic_positions[0]
    prefix = expected[:variadic_idx]
    suffix = expected[variadic_idx + 1 :]
    if len(actual) < len(prefix) + len(suffix):
        return
    for exp_dim, act_dim in zip(prefix, actual[: len(prefix)], strict=True):
        if isinstance(exp_dim, str):
            subst.setdefault(exp_dim, act_dim)
    if suffix:
        actual_suffix = actual[-len(suffix) :]
        for exp_dim, act_dim in zip(suffix, actual_suffix, strict=True):
            if isinstance(exp_dim, str):
                subst.setdefault(exp_dim, act_dim)
    variadic_name = expected[variadic_idx]
    assert isinstance(variadic_name, str)
    subst.setdefault(
        variadic_name, actual[len(prefix) : len(actual) - len(suffix) if suffix else len(actual)]
    )


def _collect_inline_dim_subst_for_param(
    *,
    param: AxonParam,
    actual_expr: AxonExpr,
    subst: dict[str, int | str | DimExprBinary | tuple[int | str | DimExprBinary, ...]],
) -> None:
    expected_type = param.type_expr
    actual_type = actual_expr.inferred_type
    if expected_type is None:
        return
    if param.optional and not isinstance(expected_type, TypeOptional):
        expected_type = TypeOptional(inner=expected_type)

    def _collect(expected: object, actual: object) -> None:
        if isinstance(expected, TypeOptional):
            if isinstance(actual, TypeOptional):
                _collect(expected.inner, actual.inner)
            else:
                _collect(expected.inner, actual)
            return
        if isinstance(actual, TypeOptional):
            _collect(expected, actual.inner)
            return
        if isinstance(expected, TypeDim):
            dim_token = _expr_to_dim_token_inline(actual_expr)
            if dim_token is not None:
                subst.setdefault(param.name, dim_token)
            return
        if isinstance(expected, TypeTensor):
            actual_dims = actual.dims if isinstance(actual, TypeTensor) else None
            if actual_dims is not None:
                _bind_inline_dim_tokens(expected.dims, actual_dims, subst=subst)
            return
        if isinstance(expected, TypeList) and isinstance(actual, TypeList):
            _collect(expected.item, actual.item)
            return
        if (
            isinstance(expected, TypeTuple)
            and isinstance(actual, TypeTuple)
            and len(expected.items) == len(actual.items)
        ):
            for expected_item, actual_item in zip(expected.items, actual.items, strict=True):
                _collect(expected_item, actual_item)

    _collect(expected_type, actual_type)


def _dim_token_to_expr_inline(
    token: int | str | DimExprBinary,
    *,
    actuals: Mapping[str, AxonExpr],
    caller_dim_names: set[str],
) -> AxonExpr | None:
    if isinstance(token, int):
        return AxonExprInt(
            value=token, inferred_type=TypeDim(), inferred_arity=1, inferred_dims=None
        )
    if isinstance(token, str):
        actual = actuals.get(token)
        if actual is not None:
            return actual
        if token in caller_dim_names:
            return AxonExprName(
                name=token, inferred_type=TypeDim(), inferred_arity=1, inferred_dims=None
            )
        return None
    left = _dim_token_to_expr_inline(token.left, actuals=actuals, caller_dim_names=caller_dim_names)
    right = _dim_token_to_expr_inline(
        token.right, actuals=actuals, caller_dim_names=caller_dim_names
    )
    if left is None or right is None:
        return None
    return AxonExprBinary(
        op=token.op,
        left=left,
        right=right,
        inferred_type=TypeDim(),
        inferred_arity=1,
        inferred_dims=None,
    )


def _call_inline_dim_subst(
    module: AxonDefinition,
    call: AxonExprCall,
    *,
    caller_dim_names: set[str],
    protected_names: set[str] | None = None,
) -> dict[str, AxonExpr]:
    actuals = _call_actual_by_param(module, call)
    raw_subst: dict[str, int | str | DimExprBinary | tuple[int | str | DimExprBinary, ...]] = {}
    for param in module.params:
        actual_expr = actuals.get(param.name)
        if actual_expr is None:
            continue
        _collect_inline_dim_subst_for_param(param=param, actual_expr=actual_expr, subst=raw_subst)
    out: dict[str, AxonExpr] = {}
    protected = protected_names or set()
    for name, token in raw_subst.items():
        if name in protected:
            continue
        if isinstance(token, tuple):
            continue
        converted = _dim_token_to_expr_inline(
            token, actuals=actuals, caller_dim_names=caller_dim_names
        )
        if converted is None:
            continue
        if name not in actuals:
            out[name] = converted
    return out


def _can_inline_module_at_call(
    module: AxonDefinition,
    call: AxonExprCall,
    *,
    caller_dim_names: set[str],
    protected_names: set[str] | None = None,
) -> bool:
    protected = protected_names or set()
    dim_actuals = _call_inline_dim_subst(
        module,
        call,
        caller_dim_names=caller_dim_names,
        protected_names=protected,
    )
    unresolved = _module_dim_names(module) - set(dim_actuals) - caller_dim_names - protected
    return not unresolved


def _substitute_stmts(
    statements: tuple[AxonStatement, ...], subst: Mapping[str, AxonExpr]
) -> tuple[AxonStatement, ...]:
    with _opt_debug_time("substitute_stmts"):
        rewritten: list[AxonStatement] = []
        current_subst = dict(subst)
        for stmt in statements:
            if isinstance(stmt, AxonBind):
                rewritten_expr = _substitute_expr(stmt.expr, current_subst)
                for target in stmt.targets:
                    current_subst.pop(target, None)
                rewritten.append(replace(stmt, expr=rewritten_expr))
            elif isinstance(stmt, AxonReturn | AxonYield):
                rewritten.append(
                    replace(
                        stmt,
                        values=tuple(
                            _substitute_expr(value, current_subst, for_return=True)
                            for value in stmt.values
                        ),
                    )
                )
            elif isinstance(stmt, AxonCond):
                rewritten.append(
                    replace(
                        stmt,
                        cond=_substitute_expr(stmt.cond, current_subst),
                        true_body=_substitute_stmts(stmt.true_body, dict(current_subst)),
                        false_body=_substitute_stmts(stmt.false_body, dict(current_subst)),
                    )
                )
            elif isinstance(stmt, AxonRepeat):
                loop_subst = dict(current_subst)
                loop_subst.pop(stmt.var, None)
                for name in stmt.targets or ():
                    loop_subst.pop(name, None)
                for name in stmt.carry or ():
                    loop_subst.pop(name, None)
                rewritten.append(
                    replace(
                        stmt,
                        from_expr=_substitute_expr(stmt.from_expr, current_subst),
                        to_expr=_substitute_expr(stmt.to_expr, current_subst),
                        step_expr=_substitute_expr(stmt.step_expr, current_subst),
                        body=_substitute_stmts(stmt.body, loop_subst),
                    )
                )
            elif isinstance(stmt, AxonScopeBind):
                scope_subst = dict(current_subst)
                for target in stmt.targets:
                    scope_subst.pop(target, None)
                rewritten.append(
                    replace(
                        stmt,
                        prefix=_substitute_path(stmt.prefix, current_subst),
                        body=_substitute_stmts(stmt.body, scope_subst),
                        kwargs={
                            key: _substitute_expr(value, current_subst)
                            if isinstance(value, AxonExpr)
                            else value
                            for key, value in stmt.kwargs.items()
                        },
                    )
                )
        return tuple(rewritten)


def _literal_from_template(value: object, template: AxonExpr) -> AxonExpr:
    if isinstance(value, bool):
        return AxonExprBool(
            value=value,
            inferred_type=template.inferred_type or TypeBool(),
            inferred_arity=1,
            inferred_dims=None,
        )
    if value is None:
        return AxonExprNull(
            inferred_type=template.inferred_type,
            inferred_arity=1,
            inferred_dims=None,
        )
    if isinstance(value, int):
        return AxonExprInt(
            value=value,
            inferred_type=template.inferred_type,
            inferred_arity=1,
            inferred_dims=template.inferred_dims,
        )
    if isinstance(value, float):
        return AxonExprFloat(
            value=value,
            inferred_type=template.inferred_type,
            inferred_arity=1,
            inferred_dims=template.inferred_dims,
        )
    raise TypeError(f"unsupported folded literal {value!r}")


def _expr_known_non_null(expr: AxonExpr) -> bool:
    inner = _unwrap_expr(expr)
    if isinstance(inner, AxonExprNull):
        return False
    if isinstance(inner, AxonExprInt | AxonExprFloat | AxonExprBool | AxonExprString | AxonExprPath):
        return True
    inferred_type = expr.inferred_type or inner.inferred_type
    return inferred_type is not None and not isinstance(inferred_type, TypeAny | TypeNull | TypeOptional)


def _fold_binary(expr: AxonExprBinary) -> AxonExpr:
    left = _unwrap_expr(expr.left)
    right = _unwrap_expr(expr.right)
    if expr.op == "or":
        if isinstance(left, AxonExprBool):
            return _literal_from_template(True, expr) if left.value else expr.right
        if isinstance(right, AxonExprBool):
            return _literal_from_template(True, expr) if right.value else expr.left
        if _folded_expr_eq(expr.left, expr.right):
            return expr.left
    if expr.op == "and":
        if isinstance(left, AxonExprBool):
            return expr.right if left.value else _literal_from_template(False, expr)
        if isinstance(right, AxonExprBool):
            return expr.left if right.value else _literal_from_template(False, expr)
        if _folded_expr_eq(expr.left, expr.right):
            return expr.left
    if expr.op == "==" and _folded_expr_eq(expr.left, expr.right):
        return _literal_from_template(True, expr)
    if expr.op == "!=" and _folded_expr_eq(expr.left, expr.right):
        return _literal_from_template(False, expr)
    if isinstance(left, AxonExprNull) and _expr_known_non_null(expr.right):
        if expr.op == "==":
            return _literal_from_template(False, expr)
        if expr.op == "!=":
            return _literal_from_template(True, expr)
    if isinstance(right, AxonExprNull) and _expr_known_non_null(expr.left):
        if expr.op == "==":
            return _literal_from_template(False, expr)
        if expr.op == "!=":
            return _literal_from_template(True, expr)
    if isinstance(left, AxonExprInt) and isinstance(right, AxonExprInt):
        if expr.op == "+":
            return _literal_from_template(left.value + right.value, expr)
        if expr.op == "-":
            return _literal_from_template(left.value - right.value, expr)
        if expr.op == "*":
            return _literal_from_template(left.value * right.value, expr)
        if expr.op == "/" and right.value != 0 and left.value % right.value == 0:
            return _literal_from_template(left.value // right.value, expr)
        if expr.op == "==":
            return _literal_from_template(left.value == right.value, expr)
        if expr.op == "!=":
            return _literal_from_template(left.value != right.value, expr)
        if expr.op == "<":
            return _literal_from_template(left.value < right.value, expr)
        if expr.op == "<=":
            return _literal_from_template(left.value <= right.value, expr)
        if expr.op == ">":
            return _literal_from_template(left.value > right.value, expr)
        if expr.op == ">=":
            return _literal_from_template(left.value >= right.value, expr)
    if isinstance(left, AxonExprFloat) and isinstance(right, AxonExprFloat):
        if expr.op == "+":
            return _literal_from_template(left.value + right.value, expr)
        if expr.op == "-":
            return _literal_from_template(left.value - right.value, expr)
        if expr.op == "*":
            return _literal_from_template(left.value * right.value, expr)
        if expr.op == "/" and right.value != 0:
            return _literal_from_template(left.value / right.value, expr)
    if isinstance(left, AxonExprBool) and isinstance(right, AxonExprBool):
        if expr.op == "and":
            return _literal_from_template(left.value and right.value, expr)
        if expr.op == "or":
            return _literal_from_template(left.value or right.value, expr)
        if expr.op == "==":
            return _literal_from_template(left.value == right.value, expr)
        if expr.op == "!=":
            return _literal_from_template(left.value != right.value, expr)
    if isinstance(left, AxonExprNull) and isinstance(right, AxonExprNull):
        if expr.op == "==":
            return _literal_from_template(True, expr)
        if expr.op == "!=":
            return _literal_from_template(False, expr)
    return expr


def _safe_fold_binary(expr: AxonExprBinary) -> AxonExpr:
    left = _unwrap_expr(expr.left)
    right = _unwrap_expr(expr.right)
    if isinstance(left, AxonExprInt) and isinstance(right, AxonExprInt):
        if expr.op == "+":
            return _literal_from_template(left.value + right.value, expr)
        if expr.op == "-":
            return _literal_from_template(left.value - right.value, expr)
        if expr.op == "*":
            return _literal_from_template(left.value * right.value, expr)
        if expr.op == "/" and right.value != 0 and left.value % right.value == 0:
            return _literal_from_template(left.value // right.value, expr)
        if expr.op == "==":
            return _literal_from_template(left.value == right.value, expr)
        if expr.op == "!=":
            return _literal_from_template(left.value != right.value, expr)
        if expr.op == "<":
            return _literal_from_template(left.value < right.value, expr)
        if expr.op == "<=":
            return _literal_from_template(left.value <= right.value, expr)
        if expr.op == ">":
            return _literal_from_template(left.value > right.value, expr)
        if expr.op == ">=":
            return _literal_from_template(left.value >= right.value, expr)
    if isinstance(left, AxonExprFloat) and isinstance(right, AxonExprFloat):
        if expr.op == "+":
            return _literal_from_template(left.value + right.value, expr)
        if expr.op == "-":
            return _literal_from_template(left.value - right.value, expr)
        if expr.op == "*":
            return _literal_from_template(left.value * right.value, expr)
        if expr.op == "/" and right.value != 0.0:
            return _literal_from_template(left.value / right.value, expr)
    if isinstance(left, AxonExprBool) and isinstance(right, AxonExprBool):
        if expr.op == "and":
            return _literal_from_template(left.value and right.value, expr)
        if expr.op == "or":
            return _literal_from_template(left.value or right.value, expr)
        if expr.op == "==":
            return _literal_from_template(left.value == right.value, expr)
        if expr.op == "!=":
            return _literal_from_template(left.value != right.value, expr)
    if isinstance(left, AxonExprNull) and isinstance(right, AxonExprNull):
        if expr.op == "==":
            return _literal_from_template(True, expr)
        if expr.op == "!=":
            return _literal_from_template(False, expr)
    if expr.op == "or":
        if isinstance(left, AxonExprBool) and left.value and _is_atomic_expr(expr.right):
            return _literal_from_template(True, expr)
        if isinstance(left, AxonExprBool) and not left.value and _is_atomic_expr(expr.right):
            return expr.right
        if isinstance(right, AxonExprBool) and right.value and _is_atomic_expr(expr.left):
            return _literal_from_template(True, expr)
        if isinstance(right, AxonExprBool) and not right.value and _is_atomic_expr(expr.left):
            return expr.left
        if _is_atomic_expr(expr.left) and _folded_expr_eq(expr.left, expr.right):
            return expr.left
    if expr.op == "and":
        if isinstance(left, AxonExprBool) and left.value and _is_atomic_expr(expr.right):
            return expr.right
        if isinstance(left, AxonExprBool) and not left.value and _is_atomic_expr(expr.right):
            return _literal_from_template(False, expr)
        if isinstance(right, AxonExprBool) and right.value and _is_atomic_expr(expr.left):
            return expr.left
        if isinstance(right, AxonExprBool) and not right.value and _is_atomic_expr(expr.left):
            return _literal_from_template(False, expr)
        if _is_atomic_expr(expr.left) and _folded_expr_eq(expr.left, expr.right):
            return expr.left
    if expr.op == "==" and _is_atomic_expr(expr.left) and _folded_expr_eq(expr.left, expr.right):
        return _literal_from_template(True, expr)
    if expr.op == "!=" and _is_atomic_expr(expr.left) and _folded_expr_eq(expr.left, expr.right):
        return _literal_from_template(False, expr)
    return expr


def _safe_fold_expr(expr: AxonExpr) -> AxonExpr:
    if isinstance(expr, AxonExprAscribe):
        return replace(expr, expr=_safe_fold_expr(expr.expr))
    if isinstance(expr, AxonExprParen):
        return replace(expr, inner=_safe_fold_expr(expr.inner))
    if isinstance(expr, AxonExprList | AxonExprTuple):
        return replace(expr, items=tuple(_safe_fold_expr(item) for item in expr.items))
    if isinstance(expr, AxonExprCall):
        return replace(
            expr,
            args=tuple(_safe_fold_expr(arg) for arg in expr.args),
            kwargs={
                key: _safe_fold_expr(value) if isinstance(value, AxonExpr) else value
                for key, value in expr.kwargs.items()
            },
        )
    if isinstance(expr, AxonExprBinary):
        return _safe_fold_binary(
            replace(expr, left=_safe_fold_expr(expr.left), right=_safe_fold_expr(expr.right))
        )
    if isinstance(expr, AxonExprTernary | AxonExprIf):
        folded = replace(
            expr,
            cond=_safe_fold_expr(expr.cond),
            true_expr=_safe_fold_expr(expr.true_expr),
            false_expr=_safe_fold_expr(expr.false_expr),
        )
        cond = _unwrap_expr(folded.cond)
        if isinstance(cond, AxonExprBool):
            selected = folded.true_expr if cond.value else folded.false_expr
            if _is_atomic_expr(selected):
                return selected
        return folded
    if isinstance(expr, AxonExprBind):
        return replace(expr, value=_safe_fold_expr(expr.value), body=_safe_fold_expr(expr.body))
    if isinstance(expr, AxonExprLambda):
        return replace(expr, body=_safe_fold_expr(expr.body))
    if isinstance(expr, AxonExprDo):
        return replace(expr, body=_safe_fold_statements(expr.body))
    if isinstance(expr, AxonExprPipe):
        return replace(
            expr,
            value=_safe_fold_expr(expr.value),
            stages=tuple(_safe_fold_expr(stage) for stage in expr.stages),
        )
    return expr


def _safe_fold_statements(statements: tuple[AxonStatement, ...]) -> tuple[AxonStatement, ...]:
    folded: list[AxonStatement] = []
    for stmt in statements:
        if isinstance(stmt, AxonBind):
            folded.append(replace(stmt, expr=_safe_fold_expr(stmt.expr)))
        elif isinstance(stmt, AxonReturn | AxonYield):
            folded.append(replace(stmt, values=tuple(_safe_fold_expr(value) for value in stmt.values)))
        elif isinstance(stmt, AxonCond):
            folded.append(
                replace(
                    stmt,
                    cond=_safe_fold_expr(stmt.cond),
                    true_body=_safe_fold_statements(stmt.true_body),
                    false_body=_safe_fold_statements(stmt.false_body),
                )
            )
        elif isinstance(stmt, AxonRepeat):
            folded.append(
                replace(
                    stmt,
                    to_expr=_safe_fold_expr(stmt.to_expr),
                    from_expr=_safe_fold_expr(stmt.from_expr),
                    step_expr=_safe_fold_expr(stmt.step_expr),
                    body=_safe_fold_statements(stmt.body),
                )
            )
        elif isinstance(stmt, AxonScopeBind):
            folded.append(
                replace(
                    stmt,
                    prefix=_safe_fold_expr(stmt.prefix),
                    kwargs={
                        key: _safe_fold_expr(value) if isinstance(value, AxonExpr) else value
                        for key, value in stmt.kwargs.items()
                    },
                    body=_safe_fold_statements(stmt.body),
                )
            )
        else:
            folded.append(stmt)
    return tuple(folded)


def _is_generated_local_name(name: str) -> bool:
    return name.startswith("__") or re.fullmatch(r"_v[0-9]+", name) is not None


def _promote_return_alias_names(
    statements: tuple[AxonStatement, ...],
) -> tuple[AxonStatement, ...]:
    counts: dict[str, int] = {}
    _count_name_uses_stmts(statements, counts)
    return_names = _return_position_names(statements)
    bind_counts: dict[str, int] = {}
    for stmt in statements:
        if isinstance(stmt, AxonBind):
            for target in stmt.targets:
                bind_counts[target] = bind_counts.get(target, 0) + 1

    rewritten: list[AxonStatement] = []
    pending_by_target: dict[str, int] = {}
    changed = False
    for stmt in statements:
        if isinstance(stmt, AxonBind) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            expr = _unwrap_expr(stmt.expr)
            if (
                isinstance(expr, AxonExprName)
                and target in return_names
                and not _is_generated_local_name(target)
                and _is_generated_local_name(expr.name)
                and counts.get(target, 0) == 1
                and counts.get(expr.name, 0) == 1
                and bind_counts.get(target, 0) == 1
                and bind_counts.get(expr.name, 0) == 1
                and expr.name in pending_by_target
            ):
                producer_idx = pending_by_target.pop(expr.name)
                producer = rewritten[producer_idx]
                if isinstance(producer, AxonBind) and target not in _expr_names(producer.expr):
                    rewritten[producer_idx] = replace(producer, targets=(target,))
                    changed = True
                    continue
        rewritten.append(stmt)
        if isinstance(stmt, AxonBind) and len(stmt.targets) == 1 and stmt.targets[0] != "_":
            pending_by_target[stmt.targets[0]] = len(rewritten) - 1
    return tuple(rewritten) if changed else statements


def _optimize_safe_statements(statements: tuple[AxonStatement, ...]) -> tuple[AxonStatement, ...]:
    folded = _safe_fold_statements(statements)
    promoted = _promote_return_alias_names(folded)
    return _inline_atomic_alias_statements(promoted)


def _optimize_safe_module(module: AxonDefinition) -> AxonDefinition:
    return replace(module, statements=_optimize_safe_statements(module.statements))


def _module_alias_expr(module: AxonDefinition) -> AxonExpr | None:
    def _unwrap_name(expr: AxonExpr) -> str | None:
        current = expr
        while isinstance(current, AxonExprAscribe | AxonExprParen):
            if isinstance(current, AxonExprAscribe):
                current = current.expr
            else:
                current = current.inner
        if isinstance(current, AxonExprName):
            return current.name
        return None

    def _is_flat_inline_expr(expr: AxonExpr) -> bool:
        inner = _unwrap_expr(expr)
        if _is_atomic_expr(inner):
            return True
        if isinstance(inner, AxonExprBinary):
            return _is_atomic_expr(inner.left) and _is_atomic_expr(inner.right)
        if isinstance(inner, AxonExprTernary):
            return (
                _is_atomic_expr(inner.cond)
                and _is_flat_inline_expr(inner.true_expr)
                and _is_flat_inline_expr(inner.false_expr)
            )
        if isinstance(inner, AxonExprCall):
            return all(_is_atomic_expr(arg) for arg in inner.args) and all(
                _is_atomic_expr(value)
                for value in inner.kwargs.values()
                if isinstance(value, AxonExpr)
            )
        if isinstance(inner, AxonExprList | AxonExprTuple):
            return all(_is_atomic_expr(item) for item in inner.items)
        if isinstance(inner, AxonExprAscribe):
            return _is_flat_inline_expr(inner.expr)
        return False

    def _synthesize_straight_line_expr(statements: tuple[AxonStatement, ...]) -> AxonExpr | None:
        if (
            not statements
            or not isinstance(statements[-1], AxonReturn)
            or len(statements[-1].values) != 1
        ):
            return None
        env: dict[str, AxonExpr] = {}
        for stmt in statements[:-1]:
            if not isinstance(stmt, AxonBind) or len(stmt.targets) != 1 or stmt.targets[0] == "_":
                return None
            env[stmt.targets[0]] = _substitute_expr(stmt.expr, env)
        result = _substitute_expr(statements[-1].values[0], env)
        if _is_flat_inline_expr(result):
            return result
        return None

    candidate: AxonExpr | None
    if len(module.statements) == 1:
        only = module.statements[0]
        if isinstance(only, AxonReturn) and len(only.values) == 1:
            candidate = only.values[0]
        else:
            return None
    elif len(module.statements) == 2:
        bind, ret = module.statements
        if (
            isinstance(bind, AxonBind)
            and len(bind.targets) == 1
            and isinstance(ret, AxonReturn)
            and len(ret.values) == 1
            and _unwrap_name(ret.values[0]) == bind.targets[0]
        ):
            candidate = bind.expr
        else:
            candidate = _synthesize_straight_line_expr(module.statements)
            if candidate is None:
                return None
    else:
        candidate = _synthesize_straight_line_expr(module.statements)
        if candidate is None:
            return None
    allowed_names = set(module.path_params)
    if module.path_param is not None:
        allowed_names.add(module.path_param)
    allowed_names.update(param.name for param in module.params)
    if not _expr_names(candidate) <= allowed_names:
        return None
    if isinstance(candidate, AxonExprCall) and candidate.callee == module.name:
        return None
    return candidate


def _inline_alias_expr(
    expr: AxonExpr,
    *,
    alias_modules: dict[str, tuple[AxonDefinition, AxonExpr]],
    active: frozenset[str] = frozenset(),
) -> AxonExpr:
    if (
        not isinstance(expr, AxonExprCall)
        or expr.callee not in alias_modules
        or expr.callee in active
    ):
        return expr
    module, alias_expr = alias_modules[expr.callee]
    env: dict[str, AxonExpr] = {}
    positional = list(expr.args)
    for name, arg in zip(module.path_params, positional[: len(module.path_params)], strict=False):
        env[name] = arg
    if module.path_param is not None and positional:
        env[module.path_param] = positional[0]
    param_args = positional[len(module.path_params) :]
    if module.path_param is not None and not module.path_params and param_args:
        param_args = param_args[1:]
    for param, arg in zip(module.params, param_args, strict=False):
        env[param.name] = arg
    for key, value in expr.kwargs.items():
        if isinstance(value, AxonExpr):
            env[key] = value
    inlined = _substitute_expr(alias_expr, env)
    return _rewrite_expr(inlined, alias_modules=alias_modules, active=active | {expr.callee})


def _rewrite_expr(
    expr: AxonExpr,
    *,
    alias_modules: dict[str, tuple[AxonDefinition, AxonExpr]],
    active: frozenset[str] = frozenset(),
) -> AxonExpr:
    if isinstance(expr, AxonExprBinary):
        expr = replace(
            expr,
            left=_rewrite_expr(expr.left, alias_modules=alias_modules, active=active),
            right=_rewrite_expr(expr.right, alias_modules=alias_modules, active=active),
        )
        return _fold_binary(expr)
    if isinstance(expr, AxonExprCall):
        expr = replace(
            expr,
            args=tuple(
                _rewrite_expr(arg, alias_modules=alias_modules, active=active) for arg in expr.args
            ),
            kwargs={
                key: _rewrite_expr(value, alias_modules=alias_modules, active=active)
                if isinstance(value, AxonExpr)
                else value
                for key, value in expr.kwargs.items()
            },
        )
        return _inline_alias_expr(expr, alias_modules=alias_modules, active=active)
    if isinstance(expr, AxonExprIf | AxonExprTernary):
        expr = replace(
            expr,
            cond=_rewrite_expr(expr.cond, alias_modules=alias_modules, active=active),
            true_expr=_rewrite_expr(expr.true_expr, alias_modules=alias_modules, active=active),
            false_expr=_rewrite_expr(expr.false_expr, alias_modules=alias_modules, active=active),
        )
        cond_value = _unwrap_expr(expr.cond)
        if isinstance(cond_value, AxonExprBool):
            return expr.true_expr if cond_value.value else expr.false_expr
        if _folded_expr_eq(expr.true_expr, expr.false_expr):
            return expr.true_expr
        return expr
    if isinstance(expr, AxonExprBind):
        return replace(
            expr,
            value=_rewrite_expr(expr.value, alias_modules=alias_modules, active=active),
            body=_rewrite_expr(expr.body, alias_modules=alias_modules, active=active),
        )
    if isinstance(expr, AxonExprDo):
        return replace(expr, body=_rewrite_statements(expr.body, alias_modules=alias_modules))
    if isinstance(expr, AxonExprLambda):
        return replace(
            expr, body=_rewrite_expr(expr.body, alias_modules=alias_modules, active=active)
        )
    if isinstance(expr, AxonExprAscribe):
        inner = _rewrite_expr(expr.expr, alias_modules=alias_modules, active=active)
        if _is_atomic_expr(inner):
            return replace(expr, expr=inner)
        return replace(expr, expr=inner)
    if isinstance(expr, AxonExprList | AxonExprTuple):
        return replace(
            expr,
            items=tuple(
                _rewrite_expr(item, alias_modules=alias_modules, active=active)
                for item in expr.items
            ),
        )
    if isinstance(expr, AxonExprParen):
        return replace(
            expr, inner=_rewrite_expr(expr.inner, alias_modules=alias_modules, active=active)
        )
    if isinstance(expr, AxonExprPipe):
        return replace(
            expr,
            value=_rewrite_expr(expr.value, alias_modules=alias_modules, active=active),
            stages=tuple(
                _rewrite_expr(item, alias_modules=alias_modules, active=active)
                for item in expr.stages
            ),
        )
    return expr


def _is_pure_expr(expr: AxonExpr) -> bool:
    if _is_atomic_expr(expr):
        return True
    if isinstance(expr, AxonExprAscribe):
        return _is_pure_expr(expr.expr)
    if isinstance(expr, AxonExprList | AxonExprTuple):
        return all(_is_pure_expr(item) for item in expr.items)
    if isinstance(expr, AxonExprBinary):
        return _is_pure_expr(expr.left) and _is_pure_expr(expr.right)
    if isinstance(expr, AxonExprTernary):
        return (
            _is_pure_expr(expr.cond)
            and _is_pure_expr(expr.true_expr)
            and _is_pure_expr(expr.false_expr)
        )
    if isinstance(expr, AxonExprCall):
        return all(_is_pure_expr(arg) for arg in expr.args) and all(
            _is_pure_expr(value) for value in expr.kwargs.values() if isinstance(value, AxonExpr)
        )
    return False


def _inline_atomic_alias_statements(
    statements: tuple[AxonStatement, ...],
) -> tuple[AxonStatement, ...]:
    with _opt_debug_time("inline_atomic_alias_statements"):
        counts: dict[str, int] = {}
        _count_name_uses_stmts(statements, counts)
        return_names = _return_position_names(statements)
        subst: dict[str, AxonExpr] = {}
        rewritten: list[AxonStatement] = []
        for stmt in statements:
            current = _substitute_stmts((stmt,), subst)[0]
            if (
                isinstance(current, AxonBind)
                and len(current.targets) == 1
                and current.targets[0] != "_"
                and _is_atomic_expr(current.expr)
                and counts.get(current.targets[0], 0) <= 1
                and (isinstance(current.expr, AxonExprName) or current.targets[0] not in return_names)
            ):
                subst[current.targets[0]] = current.expr
                continue
            rewritten.append(current)
            if isinstance(current, AxonBind):
                for target in current.targets:
                    subst.pop(target, None)
        return tuple(rewritten)


def _expr_has_flat_shape(expr: AxonExpr) -> bool:
    inner = _unwrap_expr(expr)
    if _is_atomic_expr(inner):
        return True
    if isinstance(inner, AxonExprCall):
        return all(_is_atomic_expr(arg) for arg in inner.args) and all(
            _is_atomic_expr(value) for value in inner.kwargs.values() if isinstance(value, AxonExpr)
        )
    if isinstance(inner, AxonExprBinary):
        return _is_atomic_expr(inner.left) and _is_atomic_expr(inner.right)
    if isinstance(inner, AxonExprTernary):
        return (
            _is_atomic_expr(inner.cond)
            and _expr_has_flat_shape(inner.true_expr)
            and _expr_has_flat_shape(inner.false_expr)
        )
    if isinstance(inner, AxonExprList | AxonExprTuple):
        return all(_expr_has_flat_shape(item) for item in inner.items)
    if isinstance(inner, AxonExprAscribe):
        return _expr_has_flat_shape(inner.expr)
    return False


def _stmt_has_flat_shape(stmt: AxonStatement) -> bool:
    if isinstance(stmt, AxonBind):
        return _expr_has_flat_shape(stmt.expr)
    if isinstance(stmt, AxonReturn | AxonYield):
        return all(_is_atomic_expr(value) for value in stmt.values)
    return True


def _name_used_as_call_arg_expr(expr: AxonExpr, name: str) -> bool:
    inner = _unwrap_expr(expr)
    if isinstance(inner, AxonExprCall):
        for arg in inner.args:
            arg_inner = _unwrap_expr(arg)
            if isinstance(arg_inner, AxonExprName) and arg_inner.name == name:
                return True
        for value in inner.kwargs.values():
            if not isinstance(value, AxonExpr):
                continue
            value_inner = _unwrap_expr(value)
            if isinstance(value_inner, AxonExprName) and value_inner.name == name:
                return True
        return any(_name_used_as_call_arg_expr(arg, name) for arg in inner.args) or any(
            _name_used_as_call_arg_expr(value, name)
            for value in inner.kwargs.values()
            if isinstance(value, AxonExpr)
        )
    if isinstance(inner, AxonExprBinary):
        return _name_used_as_call_arg_expr(inner.left, name) or _name_used_as_call_arg_expr(
            inner.right, name
        )
    if isinstance(inner, AxonExprTernary):
        return (
            _name_used_as_call_arg_expr(inner.cond, name)
            or _name_used_as_call_arg_expr(inner.true_expr, name)
            or _name_used_as_call_arg_expr(inner.false_expr, name)
        )
    if isinstance(inner, AxonExprList | AxonExprTuple):
        return any(_name_used_as_call_arg_expr(item, name) for item in inner.items)
    if isinstance(inner, AxonExprAscribe):
        return _name_used_as_call_arg_expr(inner.expr, name)
    return False


def _name_used_as_call_arg_stmts(statements: tuple[AxonStatement, ...], name: str) -> bool:
    for stmt in statements:
        if isinstance(stmt, AxonBind):
            if _name_used_as_call_arg_expr(stmt.expr, name):
                return True
        elif isinstance(stmt, AxonReturn | AxonYield):
            if any(_name_used_as_call_arg_expr(value, name) for value in stmt.values):
                return True
    return False


def _inline_single_use_pure_binds(
    statements: tuple[AxonStatement, ...],
) -> tuple[AxonStatement, ...]:
    with _opt_debug_time("inline_single_use_pure_binds"):
        current = statements
        while True:
            counts: dict[str, int] = {}
            _count_name_uses_stmts(current, counts)
            return_names = _return_position_names(current)
            changed = False
            for idx, stmt in enumerate(current):
                if not isinstance(stmt, AxonBind) or len(stmt.targets) != 1 or stmt.targets[0] == "_":
                    continue
                target = stmt.targets[0]
                if counts.get(target, 0) > 1 or target in return_names or not _is_pure_expr(stmt.expr):
                    continue
                if not _is_atomic_expr(stmt.expr) and _name_used_as_call_arg_stmts(
                    current[idx + 1 :], target
                ):
                    continue
                substituted_tail = _substitute_stmts(current[idx + 1 :], {target: stmt.expr})
                if not all(_stmt_has_flat_shape(item) for item in substituted_tail):
                    continue
                current = current[:idx] + substituted_tail
                changed = True
                break
            if not changed:
                return current


def _atomicize_call_args_expr(expr: AxonExpr, blocked_names: set[str]) -> tuple[tuple[AxonStatement, ...], AxonExpr]:
    prefix: list[AxonStatement] = []

    def _atomicize(value: AxonExpr) -> AxonExpr:
        if _is_atomic_expr(value):
            return value
        temp = _fresh_name("arg", blocked_names)
        prefix.append(AxonBind(targets=(temp,), expr=value))
        return _as_temp_name_expr(temp, value)

    inner = _unwrap_expr(expr)
    if isinstance(inner, AxonExprCall):
        args: list[AxonExpr] = []
        for arg in inner.args:
            arg_prefix, rewritten_arg = _atomicize_call_args_expr(arg, blocked_names)
            prefix.extend(arg_prefix)
            args.append(_atomicize(rewritten_arg))
        kwargs: dict[str, AxonKwargValue] = {}
        for key, value in inner.kwargs.items():
            if not isinstance(value, AxonExpr):
                kwargs[key] = value
                continue
            value_prefix, rewritten_value = _atomicize_call_args_expr(value, blocked_names)
            prefix.extend(value_prefix)
            kwargs[key] = _atomicize(rewritten_value)
        return tuple(prefix), replace(inner, args=tuple(args), kwargs=kwargs)
    if isinstance(inner, AxonExprBinary):
        left_prefix, left = _atomicize_call_args_expr(inner.left, blocked_names)
        right_prefix, right = _atomicize_call_args_expr(inner.right, blocked_names)
        prefix.extend(left_prefix)
        prefix.extend(right_prefix)
        return tuple(prefix), replace(inner, left=left, right=right)
    if isinstance(inner, AxonExprTernary):
        cond_prefix, cond = _atomicize_call_args_expr(inner.cond, blocked_names)
        true_prefix, true_expr = _atomicize_call_args_expr(inner.true_expr, blocked_names)
        false_prefix, false_expr = _atomicize_call_args_expr(inner.false_expr, blocked_names)
        prefix.extend(cond_prefix)
        prefix.extend(true_prefix)
        prefix.extend(false_prefix)
        return tuple(prefix), replace(inner, cond=cond, true_expr=true_expr, false_expr=false_expr)
    if isinstance(inner, AxonExprList | AxonExprTuple):
        items: list[AxonExpr] = []
        for item in inner.items:
            item_prefix, rewritten_item = _atomicize_call_args_expr(item, blocked_names)
            prefix.extend(item_prefix)
            items.append(rewritten_item)
        return tuple(prefix), replace(inner, items=tuple(items))
    if isinstance(inner, AxonExprAscribe):
        inner_prefix, rewritten = _atomicize_call_args_expr(inner.expr, blocked_names)
        prefix.extend(inner_prefix)
        return tuple(prefix), replace(inner, expr=rewritten)
    return (), expr


def _atomicize_call_args_statements(statements: tuple[AxonStatement, ...]) -> tuple[AxonStatement, ...]:
    blocked = _bound_names_statements(statements)
    rewritten: list[AxonStatement] = []
    for stmt in statements:
        if isinstance(stmt, AxonBind):
            prefix, expr = _atomicize_call_args_expr(stmt.expr, blocked)
            rewritten.extend(prefix)
            rewritten.append(replace(stmt, expr=expr))
            continue
        if isinstance(stmt, AxonReturn | AxonYield):
            values: list[AxonExpr] = []
            for value in stmt.values:
                prefix, rewritten_value = _atomicize_call_args_expr(value, blocked)
                rewritten.extend(prefix)
                values.append(rewritten_value)
            rewritten.append(replace(stmt, values=tuple(values)))
            continue
        rewritten.append(stmt)
    return tuple(rewritten)


def _dead_code_eliminate_statements(
    statements: tuple[AxonStatement, ...],
) -> tuple[AxonStatement, ...]:
    with _opt_debug_time("dead_code_eliminate_statements"):
        live: set[str] = set()
        kept_rev: list[AxonStatement] = []
        for stmt in reversed(statements):
            if isinstance(stmt, AxonReturn | AxonYield):
                for value in stmt.values:
                    live.update(_expr_names(value))
                kept_rev.append(stmt)
                continue
            if isinstance(stmt, AxonBind):
                targets = {target for target in stmt.targets if target != "_"}
                used = bool(targets & live)
                if not used and targets and _is_pure_expr(stmt.expr):
                    continue
                live.difference_update(targets)
                live.update(_expr_names(stmt.expr))
                kept_rev.append(stmt)
                continue
            if isinstance(stmt, AxonCond):
                true_body = _dead_code_eliminate_statements(stmt.true_body)
                false_body = _dead_code_eliminate_statements(stmt.false_body)
                live.update(_expr_names(stmt.cond))
                kept_rev.append(replace(stmt, true_body=true_body, false_body=false_body))
                continue
            if isinstance(stmt, AxonRepeat):
                body = _dead_code_eliminate_statements(stmt.body)
                live.update(_expr_names(stmt.from_expr))
                live.update(_expr_names(stmt.to_expr))
                live.update(_expr_names(stmt.step_expr))
                live.update(_stmt_names(body))
                kept_rev.append(replace(stmt, body=body))
                continue
            if isinstance(stmt, AxonScopeBind):
                body = _dead_code_eliminate_statements(stmt.body)
                live.update(_stmt_names(body))
                for raw_value in stmt.kwargs.values():
                    if isinstance(raw_value, AxonExpr):
                        live.update(_expr_names(raw_value))
                kept_rev.append(replace(stmt, body=body))
        return tuple(reversed(kept_rev))


def _known_bool_constraints(module: AxonDefinition) -> dict[str, bool]:
    known: dict[str, bool] = {}
    for item in module.constraints or ():
        if item.guards:
            continue
        if item.relation == "is_true" and isinstance(item.left, str):
            known[item.left] = True
        elif item.relation == "is_false" and isinstance(item.left, str):
            known[item.left] = False
        elif item.relation == "=" and isinstance(item.left, str) and isinstance(item.right, bool):
            known[item.left] = item.right
    return known


def _known_literal_constraints(module: AxonDefinition) -> dict[str, AxonExpr]:
    known: dict[str, AxonExpr] = {}
    for item in module.constraints or ():
        if item.guards or item.relation != "=" or not isinstance(item.left, str):
            continue
        right = item.right
        if isinstance(right, bool):
            known[item.left] = AxonExprBool(value=right)
        elif isinstance(right, int):
            known[item.left] = AxonExprInt(value=right)
        elif isinstance(right, float):
            known[item.left] = AxonExprFloat(value=right)
        elif right is None:
            known[item.left] = AxonExprNull()
    return known


def _fold_expr_by_known_bool(expr: AxonExpr, known: Mapping[str, bool]) -> AxonExpr:
    if isinstance(expr, AxonExprTernary):
        cond = _fold_expr_by_known_bool(expr.cond, known)
        true_expr = _fold_expr_by_known_bool(expr.true_expr, known)
        false_expr = _fold_expr_by_known_bool(expr.false_expr, known)
        if isinstance(cond, AxonExprName) and cond.name in known:
            return true_expr if known[cond.name] else false_expr
        cond_unwrapped = _unwrap_expr(cond)
        if isinstance(cond_unwrapped, AxonExprBool):
            return true_expr if cond_unwrapped.value else false_expr
        return replace(expr, cond=cond, true_expr=true_expr, false_expr=false_expr)
    if isinstance(expr, AxonExprBinary):
        return replace(
            expr,
            left=_fold_expr_by_known_bool(expr.left, known),
            right=_fold_expr_by_known_bool(expr.right, known),
        )
    if isinstance(expr, AxonExprBind):
        return replace(
            expr,
            value=_fold_expr_by_known_bool(expr.value, known),
            body=_fold_expr_by_known_bool(expr.body, known),
        )
    if isinstance(expr, AxonExprCall):
        return replace(
            expr,
            args=tuple(_fold_expr_by_known_bool(arg, known) for arg in expr.args),
            kwargs={
                key: _fold_expr_by_known_bool(value, known)
                if isinstance(value, AxonExpr)
                else value
                for key, value in expr.kwargs.items()
            },
        )
    if isinstance(expr, AxonExprAscribe):
        return replace(expr, expr=_fold_expr_by_known_bool(expr.expr, known))
    if isinstance(expr, AxonExprParen):
        return replace(expr, inner=_fold_expr_by_known_bool(expr.inner, known))
    if isinstance(expr, AxonExprList | AxonExprTuple):
        return replace(
            expr, items=tuple(_fold_expr_by_known_bool(item, known) for item in expr.items)
        )
    if isinstance(expr, AxonExprIf):
        return replace(
            expr,
            cond=_fold_expr_by_known_bool(expr.cond, known),
            true_expr=_fold_expr_by_known_bool(expr.true_expr, known),
            false_expr=_fold_expr_by_known_bool(expr.false_expr, known),
        )
    if isinstance(expr, AxonExprLambda):
        return replace(expr, body=_fold_expr_by_known_bool(expr.body, known))
    if isinstance(expr, AxonExprDo):
        return replace(expr, body=_fold_statements_by_known_bool(expr.body, known))
    if isinstance(expr, AxonExprPipe):
        return replace(
            expr,
            value=_fold_expr_by_known_bool(expr.value, known),
            stages=tuple(_fold_expr_by_known_bool(item, known) for item in expr.stages),
        )
    return expr


def _fold_statements_by_known_bool(
    statements: tuple[AxonStatement, ...],
    known: Mapping[str, bool],
    known_literals: Mapping[str, AxonExpr] | None = None,
) -> tuple[AxonStatement, ...]:
    with _opt_debug_time("fold_statements_by_known_bool"):
        local_known = dict(known)
        local_literals = dict(known_literals or {})

        def _known_value(expr: AxonExpr) -> bool | None:
            inner = _unwrap_expr(expr)
            if isinstance(inner, AxonExprBool):
                return inner.value
            if isinstance(inner, AxonExprName):
                return local_known.get(inner.name)
            return None

        def _literal_expr(expr: AxonExpr) -> AxonExpr | None:
            inner = _unwrap_expr(expr)
            if isinstance(inner, AxonExprInt | AxonExprFloat | AxonExprBool | AxonExprNull):
                return inner
            return None

        out: list[AxonStatement] = []
        for stmt in statements:
            if isinstance(stmt, AxonBind):
                folded_expr = _fold_expr_by_known_bool(
                    _substitute_expr(stmt.expr, local_literals), local_known
                )
                out.append(replace(stmt, expr=folded_expr))
                if len(stmt.targets) == 1 and stmt.targets[0] != "_":
                    target = stmt.targets[0]
                    value = _known_value(folded_expr)
                    if value is None:
                        local_known.pop(target, None)
                    else:
                        local_known[target] = value
                    literal = _literal_expr(folded_expr)
                    if literal is None:
                        local_literals.pop(target, None)
                    else:
                        local_literals[target] = literal
                continue
            if isinstance(stmt, AxonReturn | AxonYield):
                out.append(
                    replace(
                        stmt,
                        values=tuple(
                            _fold_expr_by_known_bool(value, local_known) for value in stmt.values
                        ),
                    )
                )
                continue
            if isinstance(stmt, AxonCond):
                folded_cond = _fold_expr_by_known_bool(
                    _substitute_expr(stmt.cond, local_literals), local_known
                )
                cond_value = _known_value(folded_cond)
                if cond_value is not None:
                    chosen = stmt.true_body if cond_value else stmt.false_body
                    out.extend(_fold_statements_by_known_bool(chosen, local_known, local_literals))
                    continue
                out.append(
                    replace(
                        stmt,
                        cond=folded_cond,
                        true_body=_fold_statements_by_known_bool(
                            stmt.true_body, local_known, local_literals
                        ),
                        false_body=_fold_statements_by_known_bool(
                            stmt.false_body, local_known, local_literals
                        ),
                    )
                )
                continue
            if isinstance(stmt, AxonRepeat):
                out.append(
                    replace(
                        stmt,
                        from_expr=_fold_expr_by_known_bool(
                            _substitute_expr(stmt.from_expr, local_literals), local_known
                        ),
                        to_expr=_fold_expr_by_known_bool(
                            _substitute_expr(stmt.to_expr, local_literals), local_known
                        ),
                        step_expr=_fold_expr_by_known_bool(
                            _substitute_expr(stmt.step_expr, local_literals), local_known
                        ),
                        body=_fold_statements_by_known_bool(stmt.body, local_known, local_literals),
                    )
                )
                continue
            if isinstance(stmt, AxonScopeBind):
                out.append(
                    replace(
                        stmt,
                        body=_fold_statements_by_known_bool(stmt.body, local_known, local_literals),
                        kwargs={
                            key: _fold_expr_by_known_bool(
                                _substitute_expr(value, local_literals), local_known
                            )
                            if isinstance(value, AxonExpr)
                            else value
                            for key, value in stmt.kwargs.items()
                        },
                    )
                )
                continue
            out.append(stmt)
        return tuple(out)


def _is_identity_tuple_bind(targets: tuple[str, ...], expr: AxonExpr) -> bool:
    tuple_expr = _unwrap_expr(expr)
    if not isinstance(tuple_expr, AxonExprTuple) or len(tuple_expr.items) != len(targets):
        return False
    for target, item in zip(targets, tuple_expr.items, strict=True):
        unwrapped_item = _unwrap_expr(item)
        if not isinstance(unwrapped_item, AxonExprName) or unwrapped_item.name != target:
            return False
    return True


def _is_list_destructure_expr(expr: AxonExpr) -> bool:
    tp = expr.inferred_type
    while isinstance(tp, TypeOptional):
        tp = tp.inner
    return isinstance(tp, TypeList)


def _normalize_list_destructure_binds(
    statements: tuple[AxonStatement, ...],
) -> tuple[AxonStatement, ...]:
    with _opt_debug_time("normalize_list_destructure_binds"):
        used_names = _bound_names_statements(statements)
        out: list[AxonStatement] = []
        next_idx = 1

        def fresh_temp() -> str:
            nonlocal next_idx
            while True:
                candidate = f"__list_unpack_{next_idx}"
                next_idx += 1
                if candidate not in used_names:
                    used_names.add(candidate)
                    return candidate

        for stmt in statements:
            if not isinstance(stmt, AxonBind) or len(stmt.targets) <= 1:
                out.append(stmt)
                continue
            if not _is_list_destructure_expr(stmt.expr):
                out.append(stmt)
                continue
            temp_name = fresh_temp()
            out.append(AxonBind(targets=(temp_name,), expr=stmt.expr))
            for idx, target in enumerate(stmt.targets):
                out.append(
                    AxonBind(
                        targets=(target,),
                        expr=AxonExprCall(
                            callee="_list_index",
                            args=(AxonExprName(name=temp_name), AxonExprInt(value=idx)),
                            kwargs={},
                        ),
                    )
                )
            used_names.update(name for name in stmt.targets if name != "_")
        return tuple(out)


def _rewrite_statements(
    statements: tuple[AxonStatement, ...],
    *,
    alias_modules: dict[str, tuple[AxonDefinition, AxonExpr]],
) -> tuple[AxonStatement, ...]:
    with _opt_debug_time("rewrite_statements"):
        rewritten: list[AxonStatement] = []
        for stmt in statements:
            if isinstance(stmt, AxonBind):
                rewritten_expr = _rewrite_expr(stmt.expr, alias_modules=alias_modules)
                if (
                    len(stmt.targets) == 1
                    and stmt.targets[0] != "_"
                    and isinstance(rewritten_expr, AxonExprName)
                    and stmt.targets[0] == rewritten_expr.name
                ):
                    continue
                if len(stmt.targets) > 1 and _is_identity_tuple_bind(stmt.targets, rewritten_expr):
                    continue
                rewritten.append(replace(stmt, expr=rewritten_expr))
            elif isinstance(stmt, AxonReturn | AxonYield):
                rewritten.append(
                    replace(
                        stmt,
                        values=tuple(
                            _rewrite_expr(value, alias_modules=alias_modules) for value in stmt.values
                        ),
                    )
                )
            elif isinstance(stmt, AxonCond):
                rewritten.append(
                    replace(
                        stmt,
                        cond=_rewrite_expr(stmt.cond, alias_modules=alias_modules),
                        true_body=_rewrite_statements(stmt.true_body, alias_modules=alias_modules),
                        false_body=_rewrite_statements(stmt.false_body, alias_modules=alias_modules),
                    )
                )
            elif isinstance(stmt, AxonRepeat):
                rewritten.append(
                    replace(
                        stmt,
                        from_expr=_rewrite_expr(stmt.from_expr, alias_modules=alias_modules),
                        to_expr=_rewrite_expr(stmt.to_expr, alias_modules=alias_modules),
                        step_expr=_rewrite_expr(stmt.step_expr, alias_modules=alias_modules),
                        body=_rewrite_statements(stmt.body, alias_modules=alias_modules),
                    )
                )
            elif isinstance(stmt, AxonScopeBind):
                rewritten.append(
                    replace(
                        stmt,
                        prefix=_substitute_path(stmt.prefix, {}),
                        body=_rewrite_statements(stmt.body, alias_modules=alias_modules),
                        kwargs={
                            key: _rewrite_expr(value, alias_modules=alias_modules)
                            if isinstance(value, AxonExpr)
                            else value
                            for key, value in stmt.kwargs.items()
                        },
                    )
                )
        return _dead_code_eliminate_statements(
            _inline_single_use_pure_binds(_inline_atomic_alias_statements(tuple(rewritten)))
        )


def _optimize_statements_fixpoint(
    statements: tuple[AxonStatement, ...],
    *,
    alias_modules: dict[str, tuple[AxonDefinition, AxonExpr]],
) -> tuple[AxonStatement, ...]:
    with _opt_debug_time("optimize_statements_fixpoint"):
        current = statements
        while True:
            rewritten = _rewrite_statements(current, alias_modules=alias_modules)
            if ast_equal(AxonExprDo(body=current), AxonExprDo(body=rewritten)):
                return rewritten
            current = rewritten


def _bound_names_in_order(statements: tuple[AxonStatement, ...]) -> list[str]:
    names: list[str] = []
    for stmt in statements:
        if isinstance(stmt, AxonBind):
            for target in stmt.targets:
                if target != "_":
                    names.append(target)
    return names


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

    def _rename_callees_expr(expr: AxonExpr) -> AxonExpr:
        if isinstance(expr, AxonExprCall):
            return replace(
                expr,
                callee=renames.get(expr.callee, expr.callee),
                args=tuple(_rename_callees_expr(arg) for arg in expr.args),
                kwargs={
                    key: _rename_callees_expr(value) if isinstance(value, AxonExpr) else value
                    for key, value in expr.kwargs.items()
                },
            )
        if isinstance(expr, AxonExprBinary):
            return replace(
                expr, left=_rename_callees_expr(expr.left), right=_rename_callees_expr(expr.right)
            )
        if isinstance(expr, AxonExprAscribe):
            return replace(expr, expr=_rename_callees_expr(expr.expr))
        if isinstance(expr, AxonExprTernary):
            return replace(
                expr,
                cond=_rename_callees_expr(expr.cond),
                true_expr=_rename_callees_expr(expr.true_expr),
                false_expr=_rename_callees_expr(expr.false_expr),
            )
        if isinstance(expr, AxonExprList | AxonExprTuple):
            return replace(expr, items=tuple(_rename_callees_expr(item) for item in expr.items))
        return expr

    def _rename_callees_stmts(statements: tuple[AxonStatement, ...]) -> tuple[AxonStatement, ...]:
        rewritten: list[AxonStatement] = []
        for stmt in statements:
            if isinstance(stmt, AxonBind):
                rewritten.append(replace(stmt, expr=_rename_callees_expr(stmt.expr)))
            elif isinstance(stmt, AxonReturn | AxonYield):
                rewritten.append(
                    replace(
                        stmt, values=tuple(_rename_callees_expr(value) for value in stmt.values)
                    )
                )
            else:
                rewritten.append(stmt)
        return tuple(rewritten)

    return replace(
        program,
        modules=tuple(
            replace(
                module,
                name=renames.get(module.name, module.name),
                statements=_rename_callees_stmts(module.statements),
            )
            for module in program.modules
        ),
    )


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


def _module_call_graph(program: AxonFile) -> dict[str, set[str]]:
    module_names = {module.name for module in program.modules}
    graph: dict[str, set[str]] = {module.name: set() for module in program.modules}
    for module in program.modules:
        for call in _walk_stmt_calls(module.statements):
            if call.callee in module_names:
                graph[module.name].add(call.callee)
    return graph


def _module_callsites(program: AxonFile) -> dict[str, list[tuple[str, AxonExprCall]]]:
    module_names = {module.name for module in program.modules}
    callsites: dict[str, list[tuple[str, AxonExprCall]]] = {
        module.name: [] for module in program.modules
    }
    for caller in program.modules:
        for call in _walk_stmt_calls(caller.statements):
            if call.callee in module_names:
                callsites[call.callee].append((caller.name, call))
    return callsites


def _module_sccs(graph: dict[str, set[str]]) -> list[tuple[str, ...]]:
    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    sccs: list[tuple[str, ...]] = []

    def strongconnect(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlink[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for succ in graph[node]:
            if succ not in indices:
                strongconnect(succ)
                lowlink[node] = min(lowlink[node], lowlink[succ])
            elif succ in on_stack:
                lowlink[node] = min(lowlink[node], indices[succ])
        if lowlink[node] != indices[node]:
            return
        component: list[str] = []
        while stack:
            item = stack.pop()
            on_stack.remove(item)
            component.append(item)
            if item == node:
                break
        sccs.append(tuple(component))

    for node in graph:
        if node not in indices:
            strongconnect(node)
    return sccs


def _deduped_path_names(module: AxonDefinition) -> tuple[str, ...]:
    names = list(module.path_params)
    if module.path_param is not None and module.path_param not in names:
        names.append(module.path_param)
    return tuple(names)


def _constraint_operand_names(operand: object) -> set[str]:
    if isinstance(operand, str):
        return {operand}
    if isinstance(operand, DimExprBinary):
        return _constraint_operand_names(operand.left) | _constraint_operand_names(operand.right)
    if isinstance(operand, tuple):
        names: set[str] = set()
        for item in operand:
            names.update(_constraint_operand_names(item))
        return names
    return set()


def _constraint_names(constraint: Constraint) -> set[str]:
    names = _constraint_operand_names(constraint.left)
    if constraint.right is not None:
        names.update(_constraint_operand_names(constraint.right))
    for guard in constraint.guards:
        names.update(_constraint_names(guard))
    return names


def _is_trivial_identity_constraint(constraint: Constraint) -> bool:
    return (
        constraint.relation == "="
        and constraint.right is not None
        and constraint.left == constraint.right
    )


def _module_used_names(module: AxonDefinition) -> set[str]:
    used = _stmt_param_like_names(module.statements)
    for param in module.params:
        if param.default_expr is not None:
            used.update(_expr_param_like_names(param.default_expr))
    needed_dim_names = set(used)
    needed_dim_names.update(_type_dim_names(module.return_type_expr))
    for constraint in module.constraints or ():
        if _is_trivial_identity_constraint(constraint):
            continue
        constraint_names = _constraint_names(constraint)
        used.update(constraint_names)
        needed_dim_names.update(constraint_names)
    for param in module.params:
        if _type_dim_names(param.type_expr) & needed_dim_names:
            used.add(param.name)
    return used


def _param_names(module: AxonDefinition) -> tuple[str, ...]:
    names: list[str] = list(_deduped_path_names(module))
    names.extend(param.name for param in module.params)
    return tuple(names)


def _param_defaults(module: AxonDefinition) -> dict[str, AxonExpr]:
    defaults: dict[str, AxonExpr] = {}
    for param in module.params:
        if param.default_expr is not None:
            defaults[param.name] = param.default_expr
        elif param.optional:
            defaults[param.name] = AxonExprNull()
    return defaults


def _provided_call_actuals_by_param(module: AxonDefinition, call: AxonExprCall) -> dict[str, AxonExpr]:
    names = _param_names(module)
    actuals: dict[str, AxonExpr] = {}
    for name, arg in zip(names, call.args, strict=False):
        actuals[name] = arg
    for key, value in call.kwargs.items():
        if key in names and isinstance(value, AxonExpr):
            actuals[key] = value
    return actuals


def _call_actual_by_param(module: AxonDefinition, call: AxonExprCall) -> dict[str, AxonExpr]:
    actuals = _param_defaults(module)
    actuals.update(_provided_call_actuals_by_param(module, call))
    return actuals


def _rewrite_call_for_signature_change(
    call: AxonExprCall, *, old_module: AxonDefinition, new_module: AxonDefinition
) -> AxonExprCall:
    provided_actuals = _provided_call_actuals_by_param(old_module, call)
    old_names = _param_names(old_module)
    kept_names = set(_param_names(new_module))
    original_positional_names = old_names[: len(call.args)]

    new_args: list[AxonExpr] = []
    new_kwargs: dict[str, AxonKwargValue] = {
        key: value
        for key, value in call.kwargs.items()
        if key not in old_names and key not in kept_names
    }
    positional_prefix_open = True
    for name in _param_names(new_module):
        provided = provided_actuals.get(name)
        if provided is None:
            positional_prefix_open = False
            continue
        if (
            positional_prefix_open
            and len(new_args) < len(original_positional_names)
            and original_positional_names[len(new_args)] == name
        ):
            new_args.append(provided)
            continue
        positional_prefix_open = False
        new_kwargs[name] = provided
    return replace(call, args=tuple(new_args), kwargs=new_kwargs)


def _canonicalize_path_params(program: AxonFile) -> AxonFile:
    modules: list[AxonDefinition] = []
    for module in program.modules:
        path_names = _deduped_path_names(module)
        if not path_names:
            modules.append(module)
            continue
        path_params = tuple(AxonParam(name=name, type_expr=TypePath()) for name in path_names)
        modules.append(
            replace(
                module,
                path_param=None,
                path_params=(),
                params=path_params + module.params,
            )
        )
    return replace(program, modules=tuple(modules))


def _rewrite_calls_expr(
    expr: AxonExpr, specs: Mapping[str, tuple[AxonDefinition, AxonDefinition]]
) -> AxonExpr:
    if isinstance(expr, AxonExprBinary):
        return replace(
            expr,
            left=_rewrite_calls_expr(expr.left, specs),
            right=_rewrite_calls_expr(expr.right, specs),
        )
    if isinstance(expr, AxonExprBind):
        return replace(
            expr,
            value=_rewrite_calls_expr(expr.value, specs),
            body=_rewrite_calls_expr(expr.body, specs),
        )
    if isinstance(expr, AxonExprCall):
        rewritten = replace(
            expr,
            args=tuple(_rewrite_calls_expr(arg, specs) for arg in expr.args),
            kwargs={
                key: _rewrite_calls_expr(value, specs) if isinstance(value, AxonExpr) else value
                for key, value in expr.kwargs.items()
            },
        )
        spec = specs.get(rewritten.callee)
        if spec is None:
            return rewritten
        old_module, new_module = spec
        return _rewrite_call_for_signature_change(
            rewritten, old_module=old_module, new_module=new_module
        )
    if isinstance(expr, AxonExprDo):
        return replace(expr, body=_rewrite_calls_stmts(expr.body, specs))
    if isinstance(expr, AxonExprIf | AxonExprTernary):
        return replace(
            expr,
            cond=_rewrite_calls_expr(expr.cond, specs),
            true_expr=_rewrite_calls_expr(expr.true_expr, specs),
            false_expr=_rewrite_calls_expr(expr.false_expr, specs),
        )
    if isinstance(expr, AxonExprLambda):
        return replace(expr, body=_rewrite_calls_expr(expr.body, specs))
    if isinstance(expr, AxonExprAscribe):
        return replace(expr, expr=_rewrite_calls_expr(expr.expr, specs))
    if isinstance(expr, AxonExprList | AxonExprTuple):
        return replace(expr, items=tuple(_rewrite_calls_expr(item, specs) for item in expr.items))
    if isinstance(expr, AxonExprParen):
        return replace(expr, inner=_rewrite_calls_expr(expr.inner, specs))
    if isinstance(expr, AxonExprPipe):
        return replace(
            expr,
            value=_rewrite_calls_expr(expr.value, specs),
            stages=tuple(_rewrite_calls_expr(item, specs) for item in expr.stages),
        )
    return expr


def _rewrite_calls_stmts(
    statements: tuple[AxonStatement, ...], specs: Mapping[str, tuple[AxonDefinition, AxonDefinition]]
) -> tuple[AxonStatement, ...]:
    rewritten: list[AxonStatement] = []
    for stmt in statements:
        if isinstance(stmt, AxonBind):
            rewritten.append(replace(stmt, expr=_rewrite_calls_expr(stmt.expr, specs)))
        elif isinstance(stmt, AxonReturn | AxonYield):
            rewritten.append(
                replace(
                    stmt, values=tuple(_rewrite_calls_expr(value, specs) for value in stmt.values)
                )
            )
        elif isinstance(stmt, AxonCond):
            rewritten.append(
                replace(
                    stmt,
                    cond=_rewrite_calls_expr(stmt.cond, specs),
                    true_body=_rewrite_calls_stmts(stmt.true_body, specs),
                    false_body=_rewrite_calls_stmts(stmt.false_body, specs),
                )
            )
        elif isinstance(stmt, AxonRepeat):
            rewritten.append(
                replace(
                    stmt,
                    from_expr=_rewrite_calls_expr(stmt.from_expr, specs),
                    to_expr=_rewrite_calls_expr(stmt.to_expr, specs),
                    step_expr=_rewrite_calls_expr(stmt.step_expr, specs),
                    body=_rewrite_calls_stmts(stmt.body, specs),
                )
            )
        elif isinstance(stmt, AxonScopeBind):
            rewritten.append(
                replace(
                    stmt,
                    body=_rewrite_calls_stmts(stmt.body, specs),
                    kwargs={
                        key: _rewrite_calls_expr(value, specs)
                        if isinstance(value, AxonExpr)
                        else value
                        for key, value in stmt.kwargs.items()
                    },
                )
            )
    return tuple(rewritten)


def _prune_unused_module_params(program: AxonFile) -> AxonFile:
    with _opt_debug_time("prune_unused_module_params"):
        rewritten_modules: list[AxonDefinition] = []
        specs: dict[str, tuple[AxonDefinition, AxonDefinition]] = {}
        for module in program.modules:
            used = _module_used_names(module)
            new_module = replace(
                module,
                path_param=module.path_param
                if module.path_param is not None and module.path_param in used
                else None,
                path_params=tuple(name for name in module.path_params if name in used),
                params=tuple(param for param in module.params if param.name in used),
            )
            specs[module.name] = (module, new_module)
            rewritten_modules.append(new_module)
        rewritten_program = replace(program, modules=tuple(rewritten_modules))
        validate_flat_axon_file(rewritten_program)
        final_modules = tuple(
            replace(module, statements=_rewrite_calls_stmts(module.statements, specs))
            for module in rewritten_program.modules
        )
        final_program = replace(
            rewritten_program,
            modules=final_modules,
        )
        validate_flat_axon_file(final_program)
        return final_program


def _module_scope_param_name(module: AxonDefinition) -> str | None:
    for param in module.params:
        if isinstance(param.type_expr, TypePath):
            return param.name
    return None


def _loop_helper_scope(module: AxonDefinition) -> tuple[str, str] | None:
    marker = "__loop_"
    before, sep, after = module.name.partition(marker)
    del before
    if not sep:
        return None
    loop_name, sep, _rest = after.partition("_")
    if not sep or not loop_name or not module.params:
        return None
    loop_var = module.params[0].name
    return loop_name, loop_var


def _prefix_loop_path(expr: AxonExprPath, *, loop_name: str, loop_var: str) -> AxonExprPath:
    if not expr.absolute or not expr.parts:
        return expr
    prefix = (loop_name, f"{{{loop_var}}}")
    if expr.parts[: len(prefix)] == prefix:
        return expr
    if expr.parts[0].startswith("{") and expr.parts[0].endswith("}"):
        return expr
    return replace(expr, parts=(*prefix, *expr.parts))


def _prefix_loop_paths_expr(expr: AxonExpr, *, loop_name: str, loop_var: str) -> AxonExpr:
    if isinstance(expr, AxonExprPath):
        return _prefix_loop_path(expr, loop_name=loop_name, loop_var=loop_var)
    if isinstance(expr, AxonExprBinary):
        return replace(
            expr,
            left=_prefix_loop_paths_expr(expr.left, loop_name=loop_name, loop_var=loop_var),
            right=_prefix_loop_paths_expr(expr.right, loop_name=loop_name, loop_var=loop_var),
        )
    if isinstance(expr, AxonExprCall):
        if expr.callee.startswith("_config_") or expr.callee.startswith("Config."):
            return replace(
                expr,
                args=tuple(
                    arg
                    if isinstance(arg, AxonExprPath)
                    else _prefix_loop_paths_expr(arg, loop_name=loop_name, loop_var=loop_var)
                    for arg in expr.args
                ),
                kwargs={
                    key: (
                        value
                        if isinstance(value, AxonExprPath)
                        else _prefix_loop_paths_expr(value, loop_name=loop_name, loop_var=loop_var)
                    )
                    if isinstance(value, AxonExpr)
                    else value
                    for key, value in expr.kwargs.items()
                },
            )
        return replace(
            expr,
            args=tuple(
                _prefix_loop_paths_expr(arg, loop_name=loop_name, loop_var=loop_var)
                for arg in expr.args
            ),
            kwargs={
                key: _prefix_loop_paths_expr(value, loop_name=loop_name, loop_var=loop_var)
                if isinstance(value, AxonExpr)
                else value
                for key, value in expr.kwargs.items()
            },
        )
    if isinstance(expr, AxonExprIf | AxonExprTernary):
        return replace(
            expr,
            cond=_prefix_loop_paths_expr(expr.cond, loop_name=loop_name, loop_var=loop_var),
            true_expr=_prefix_loop_paths_expr(expr.true_expr, loop_name=loop_name, loop_var=loop_var),
            false_expr=_prefix_loop_paths_expr(expr.false_expr, loop_name=loop_name, loop_var=loop_var),
        )
    if isinstance(expr, AxonExprBind):
        return replace(
            expr,
            value=_prefix_loop_paths_expr(expr.value, loop_name=loop_name, loop_var=loop_var),
            body=_prefix_loop_paths_expr(expr.body, loop_name=loop_name, loop_var=loop_var),
        )
    if isinstance(expr, AxonExprDo):
        return replace(
            expr,
            body=_prefix_loop_paths_statements(expr.body, loop_name=loop_name, loop_var=loop_var),
        )
    if isinstance(expr, AxonExprLambda):
        return replace(
            expr,
            body=_prefix_loop_paths_expr(expr.body, loop_name=loop_name, loop_var=loop_var),
        )
    if isinstance(expr, AxonExprAscribe):
        return replace(
            expr,
            expr=_prefix_loop_paths_expr(expr.expr, loop_name=loop_name, loop_var=loop_var),
        )
    if isinstance(expr, AxonExprList | AxonExprTuple):
        return replace(
            expr,
            items=tuple(
                _prefix_loop_paths_expr(item, loop_name=loop_name, loop_var=loop_var)
                for item in expr.items
            ),
        )
    if isinstance(expr, AxonExprParen):
        return replace(
            expr,
            inner=_prefix_loop_paths_expr(expr.inner, loop_name=loop_name, loop_var=loop_var),
        )
    if isinstance(expr, AxonExprPipe):
        return replace(
            expr,
            value=_prefix_loop_paths_expr(expr.value, loop_name=loop_name, loop_var=loop_var),
            stages=tuple(
                _prefix_loop_paths_expr(item, loop_name=loop_name, loop_var=loop_var)
                for item in expr.stages
            ),
        )
    return expr


def _prefix_loop_paths_statements(
    statements: tuple[AxonStatement, ...], *, loop_name: str, loop_var: str
) -> tuple[AxonStatement, ...]:
    out: list[AxonStatement] = []
    for stmt in statements:
        if isinstance(stmt, AxonBind):
            out.append(
                replace(
                    stmt,
                    expr=_prefix_loop_paths_expr(stmt.expr, loop_name=loop_name, loop_var=loop_var),
                )
            )
        elif isinstance(stmt, AxonReturn | AxonYield):
            out.append(
                replace(
                    stmt,
                    values=tuple(
                        _prefix_loop_paths_expr(value, loop_name=loop_name, loop_var=loop_var)
                        for value in stmt.values
                    ),
                )
            )
        elif isinstance(stmt, AxonCond):
            out.append(
                replace(
                    stmt,
                    cond=_prefix_loop_paths_expr(stmt.cond, loop_name=loop_name, loop_var=loop_var),
                    true_body=_prefix_loop_paths_statements(
                        stmt.true_body, loop_name=loop_name, loop_var=loop_var
                    ),
                    false_body=_prefix_loop_paths_statements(
                        stmt.false_body, loop_name=loop_name, loop_var=loop_var
                    ),
                )
            )
        elif isinstance(stmt, AxonRepeat):
            out.append(
                replace(
                    stmt,
                    from_expr=_prefix_loop_paths_expr(
                        stmt.from_expr, loop_name=loop_name, loop_var=loop_var
                    ),
                    to_expr=_prefix_loop_paths_expr(
                        stmt.to_expr, loop_name=loop_name, loop_var=loop_var
                    ),
                    step_expr=_prefix_loop_paths_expr(
                        stmt.step_expr, loop_name=loop_name, loop_var=loop_var
                    ),
                    body=_prefix_loop_paths_statements(
                        stmt.body, loop_name=loop_name, loop_var=loop_var
                    ),
                )
            )
        elif isinstance(stmt, AxonScopeBind):
            out.append(
                replace(
                    stmt,
                    prefix=_prefix_loop_path(stmt.prefix, loop_name=loop_name, loop_var=loop_var),
                    body=_prefix_loop_paths_statements(
                        stmt.body, loop_name=loop_name, loop_var=loop_var
                    ),
                    kwargs={
                        key: _prefix_loop_paths_expr(value, loop_name=loop_name, loop_var=loop_var)
                        if isinstance(value, AxonExpr)
                        else value
                        for key, value in stmt.kwargs.items()
                    },
                )
            )
    return tuple(out)


def _prefix_loop_helper_paths(program: AxonFile) -> AxonFile:
    modules: list[AxonDefinition] = []
    changed = False
    for module in program.modules:
        scope = _loop_helper_scope(module)
        if scope is None:
            modules.append(module)
            continue
        loop_name, loop_var = scope
        statements = _prefix_loop_paths_statements(
            module.statements, loop_name=loop_name, loop_var=loop_var
        )
        changed = changed or statements != module.statements
        modules.append(replace(module, statements=statements))
    return replace(program, modules=tuple(modules)) if changed else program


def _is_safe_specialization_actual(expr: AxonExpr) -> bool:
    expr = _unwrap_expr(expr)
    if isinstance(expr, AxonExprPath):
        return not any(part.startswith("{") and part.endswith("}") for part in expr.parts)
    return isinstance(
        expr,
        (
            AxonExprInt,
            AxonExprFloat,
            AxonExprBool,
            AxonExprNull,
            AxonExprString,
        ),
    )


def _specialize_single_callsite_modules(program: AxonFile, *, main_module: str | None) -> AxonFile:
    with _opt_debug_time("specialize_single_callsite_modules"):
        graph = _module_call_graph(program)
        callsites = _module_callsites(program)
        sccs = _module_sccs(graph)
        scc_by_module: dict[str, tuple[str, ...]] = {}
        for component in sccs:
            for name in component:
                scc_by_module[name] = component

        rewritten_modules: list[AxonDefinition] = []
        specs: dict[str, tuple[AxonDefinition, AxonDefinition]] = {}

        for module in program.modules:
            callers = callsites.get(module.name, [])
            recursive_component = scc_by_module.get(module.name, ())
            if (
                main_module is not None
                and module.name == main_module
                or len(callers) != 1
                or len(recursive_component) > 1
                or (callers and callers[0][0] == module.name)
            ):
                specs[module.name] = (module, module)
                rewritten_modules.append(module)
                continue
            caller_name, call = callers[0]
            del caller_name
            actuals = _call_actual_by_param(module, call)
            subst = {
                name: expr
                for name, expr in actuals.items()
                if _is_safe_specialization_actual(expr)
            }
            subst = _filter_dim_safe_param_subst(module, subst)
            if not subst:
                specs[module.name] = (module, module)
                rewritten_modules.append(module)
                continue
            new_module = replace(
                module,
                params=tuple(param for param in module.params if param.name not in subst),
                statements=_substitute_stmts(module.statements, subst),
            )
            specs[module.name] = (module, new_module)
            rewritten_modules.append(new_module)

        rewritten_program = replace(program, modules=tuple(rewritten_modules))
        validate_flat_axon_file(rewritten_program, main_module=main_module)
        final_modules = tuple(
            replace(module, statements=_rewrite_calls_stmts(module.statements, specs))
            for module in rewritten_program.modules
        )
        final_program = replace(
            rewritten_program,
            modules=final_modules,
        )
        validate_flat_axon_file(final_program, main_module=main_module)
        return final_program


def _is_straight_line_module(module: AxonDefinition) -> bool:
    if not module.statements or not isinstance(module.statements[-1], AxonReturn):
        return False
    contains_control_select = False
    for stmt in module.statements:
        if isinstance(stmt, AxonCond | AxonRepeat | AxonScopeBind | AxonYield):
            return False
        if isinstance(stmt, AxonBind) and _expr_contains_control_select(stmt.expr):
            contains_control_select = True
        if isinstance(stmt, AxonReturn) and any(
            _expr_contains_control_select(value) for value in stmt.values
        ):
            contains_control_select = True
    return not contains_control_select or _is_trivial_control_wrapper_module(module)


def _is_trivial_control_wrapper_module(module: AxonDefinition) -> bool:
    if len(module.statements) < 2:
        return False
    *prefix, ret = module.statements
    if not isinstance(ret, AxonReturn) or len(ret.values) != 1:
        return False
    value = _unwrap_expr(ret.values[0])
    if not isinstance(value, AxonExprName):
        return False
    control_bind_count = 0
    control_target: str | None = None
    for stmt in prefix:
        if not isinstance(stmt, AxonBind) or len(stmt.targets) != 1:
            return False
        if not _expr_contains_control_select(stmt.expr):
            continue
        control_bind_count += 1
        control_target = stmt.targets[0]
    return control_bind_count == 1 and value.name == control_target


def _expr_contains_control_select(expr: AxonExpr) -> bool:
    inner = _unwrap_expr(expr)
    if isinstance(inner, AxonExprIf | AxonExprTernary | AxonExprDo | AxonExprLambda):
        return True
    if isinstance(inner, AxonExprCall):
        return any(_expr_contains_control_select(arg) for arg in inner.args) or any(
            _expr_contains_control_select(value)
            for value in inner.kwargs.values()
            if isinstance(value, AxonExpr)
        )
    if isinstance(inner, AxonExprBinary):
        return _expr_contains_control_select(inner.left) or _expr_contains_control_select(
            inner.right
        )
    if isinstance(inner, AxonExprList | AxonExprTuple):
        return any(_expr_contains_control_select(item) for item in inner.items)
    if isinstance(inner, AxonExprAscribe):
        return _expr_contains_control_select(inner.expr)
    if isinstance(inner, AxonExprParen):
        return _expr_contains_control_select(inner.inner)
    return False


def _inline_call_bind_statements(
    statements: tuple[AxonStatement, ...],
    *,
    caller: AxonDefinition,
    inline_modules: Mapping[str, AxonDefinition],
    protected_names: set[str] | None = None,
) -> tuple[AxonStatement, ...]:
    blocked = set(_param_names(caller))
    blocked.update(_bound_names_statements(statements))
    rewritten: list[AxonStatement] = []
    for stmt in statements:
        if not isinstance(stmt, AxonBind) or not isinstance(stmt.expr, AxonExprCall):
            rewritten.append(stmt)
            continue
        callee = inline_modules.get(stmt.expr.callee)
        if callee is None:
            rewritten.append(stmt)
            continue
        if not _can_inline_module_at_call(
            callee,
            stmt.expr,
            caller_dim_names=_module_dim_names(caller),
            protected_names=protected_names,
        ):
            rewritten.append(stmt)
            continue
        actuals = _call_actual_by_param(callee, stmt.expr)
        dim_actuals = _call_inline_dim_subst(
            callee,
            stmt.expr,
            caller_dim_names=_module_dim_names(caller),
            protected_names=protected_names,
        )
        freshened, renames = _freshen_inline_module_statements(
            callee,
            blocked_names=blocked | set(stmt.targets),
        )
        subst: dict[str, AxonExpr] = {}
        for name, value in {**dim_actuals, **actuals}.items():
            subst[renames.get(name, name)] = value
        substituted = _substitute_stmts(freshened, subst)
        inlined = _replace_returns_with_bind(substituted, targets=stmt.targets, call_expr=stmt.expr)
        allowed_free = blocked | set(_param_names(caller)) | set(stmt.targets)
        if _inline_free_names(inlined) - allowed_free:
            rewritten.append(stmt)
            continue
        blocked.update(_bound_names_statements(inlined))
        rewritten.extend(inlined)
    return tuple(rewritten)


def _call_inline_result_expr(
    callee: AxonDefinition,
    call: AxonExprCall,
    *,
    caller: AxonDefinition,
    blocked_names: set[str],
    protected_names: set[str] | None = None,
) -> tuple[tuple[AxonStatement, ...], AxonExpr] | None:
    if not _can_inline_module_at_call(
        callee,
        call,
        caller_dim_names=_module_dim_names(caller),
        protected_names=protected_names,
    ):
        return None
    actuals = _call_actual_by_param(callee, call)
    dim_actuals = _call_inline_dim_subst(
        callee,
        call,
        caller_dim_names=_module_dim_names(caller),
        protected_names=protected_names,
    )
    freshened, renames = _freshen_inline_module_statements(callee, blocked_names=blocked_names)
    subst: dict[str, AxonExpr] = {}
    for name, value in {**dim_actuals, **actuals}.items():
        subst[renames.get(name, name)] = value
    substituted = _substitute_stmts(freshened, subst)
    if not substituted:
        return None
    last = substituted[-1]
    if not isinstance(last, AxonReturn):
        return None
    prefix = substituted[:-1]
    blocked_names.update(_bound_names_statements(prefix))
    if len(last.values) == 1:
        candidate = (*prefix, AxonReturn(values=(last.values[0],)))
        if _inline_free_names(candidate) - blocked_names - set(_param_names(caller)):
            return None
        return prefix, last.values[0]
    candidate = (*prefix, last)
    if _inline_free_names(candidate) - blocked_names - set(_param_names(caller)):
        return None
    return (
        prefix,
        AxonExprTuple(
            items=last.values,
            inferred_type=call.inferred_type,
            inferred_arity=call.inferred_arity,
            inferred_dims=call.inferred_dims,
        ),
    )


def _as_temp_name_expr(name: str, template: AxonExpr) -> AxonExprName:
    return AxonExprName(
        name=name,
        inferred_type=template.inferred_type,
        inferred_arity=1 if template.inferred_arity is None else template.inferred_arity,
        inferred_dims=template.inferred_dims,
    )


def _inline_expr_position_modules_expr(
    expr: AxonExpr,
    *,
    caller: AxonDefinition,
    inline_modules: Mapping[str, AxonDefinition],
    blocked_names: set[str],
    need_atomic: bool,
    protected_names: set[str] | None = None,
) -> tuple[tuple[AxonStatement, ...], AxonExpr]:
    prefix: list[AxonStatement] = []

    def _atomicize(current: AxonExpr) -> AxonExpr:
        if not need_atomic or _is_atomic_expr(current):
            return current
        temp = _fresh_name("expr", blocked_names)
        prefix.append(AxonBind(targets=(temp,), expr=current))
        return _as_temp_name_expr(temp, current)

    if isinstance(expr, AxonExprCall):
        rewritten_args: list[AxonExpr] = []
        for arg in expr.args:
            arg_prefix, arg_expr = _inline_expr_position_modules_expr(
                arg,
                caller=caller,
                inline_modules=inline_modules,
                blocked_names=blocked_names,
                need_atomic=True,
                protected_names=protected_names,
            )
            prefix.extend(arg_prefix)
            rewritten_args.append(arg_expr)
        rewritten_kwargs: dict[str, AxonKwargValue] = {}
        for key, value in expr.kwargs.items():
            if not isinstance(value, AxonExpr):
                rewritten_kwargs[key] = value
                continue
            value_prefix, value_expr = _inline_expr_position_modules_expr(
                value,
                caller=caller,
                inline_modules=inline_modules,
                blocked_names=blocked_names,
                need_atomic=True,
                protected_names=protected_names,
            )
            prefix.extend(value_prefix)
            rewritten_kwargs[key] = value_expr
        rewritten_call = replace(expr, args=tuple(rewritten_args), kwargs=rewritten_kwargs)
        callee = inline_modules.get(rewritten_call.callee)
        if callee is not None:
            inlined = _call_inline_result_expr(
                callee,
                rewritten_call,
                caller=caller,
                blocked_names=blocked_names,
                protected_names=protected_names,
            )
            if inlined is not None:
                inline_prefix, inline_expr = inlined
                original_prefix_len = len(prefix)
                prefix.extend(inline_prefix)
                nested_prefix, nested_expr = _inline_expr_position_modules_expr(
                    inline_expr,
                    caller=caller,
                    inline_modules=inline_modules,
                    blocked_names=blocked_names,
                    need_atomic=need_atomic,
                    protected_names=protected_names,
                )
                prefix.extend(nested_prefix)
                candidate_prefix = tuple(prefix)
                allowed_free = blocked_names | set(_param_names(caller))
                allowed_free.update(_bound_names_statements(candidate_prefix))
                if (
                    _stmt_expr_names(candidate_prefix)
                    | _expr_names(nested_expr)
                ) - allowed_free:
                    prefix = prefix[:original_prefix_len]
                    return tuple(prefix), _atomicize(rewritten_call)
                return tuple(prefix), nested_expr
        return tuple(prefix), _atomicize(rewritten_call)
    if isinstance(expr, AxonExprBinary):
        left_prefix, left_expr = _inline_expr_position_modules_expr(
            expr.left,
            caller=caller,
            inline_modules=inline_modules,
            blocked_names=blocked_names,
            need_atomic=True,
            protected_names=protected_names,
        )
        right_prefix, right_expr = _inline_expr_position_modules_expr(
            expr.right,
            caller=caller,
            inline_modules=inline_modules,
            blocked_names=blocked_names,
            need_atomic=True,
            protected_names=protected_names,
        )
        prefix.extend(left_prefix)
        prefix.extend(right_prefix)
        return tuple(prefix), _atomicize(replace(expr, left=left_expr, right=right_expr))
    if isinstance(expr, AxonExprTernary):
        cond_prefix, cond_expr = _inline_expr_position_modules_expr(
            expr.cond,
            caller=caller,
            inline_modules=inline_modules,
            blocked_names=blocked_names,
            need_atomic=True,
            protected_names=protected_names,
        )
        true_prefix, true_expr = _inline_expr_position_modules_expr(
            expr.true_expr,
            caller=caller,
            inline_modules=inline_modules,
            blocked_names=blocked_names,
            need_atomic=False,
            protected_names=protected_names,
        )
        false_prefix, false_expr = _inline_expr_position_modules_expr(
            expr.false_expr,
            caller=caller,
            inline_modules=inline_modules,
            blocked_names=blocked_names,
            need_atomic=False,
            protected_names=protected_names,
        )
        prefix.extend(cond_prefix)
        prefix.extend(true_prefix)
        prefix.extend(false_prefix)
        return (
            tuple(prefix),
            _atomicize(replace(expr, cond=cond_expr, true_expr=true_expr, false_expr=false_expr)),
        )
    if isinstance(expr, AxonExprAscribe):
        inner_prefix, inner_expr = _inline_expr_position_modules_expr(
            expr.expr,
            caller=caller,
            inline_modules=inline_modules,
            blocked_names=blocked_names,
            need_atomic=need_atomic,
            protected_names=protected_names,
        )
        prefix.extend(inner_prefix)
        return tuple(prefix), replace(expr, expr=inner_expr)
    if isinstance(expr, AxonExprParen):
        inner_prefix, inner_expr = _inline_expr_position_modules_expr(
            expr.inner,
            caller=caller,
            inline_modules=inline_modules,
            blocked_names=blocked_names,
            need_atomic=need_atomic,
            protected_names=protected_names,
        )
        prefix.extend(inner_prefix)
        return tuple(prefix), replace(expr, inner=inner_expr)
    if isinstance(expr, AxonExprTuple | AxonExprList):
        rewritten_items: list[AxonExpr] = []
        for item in expr.items:
            item_prefix, item_expr = _inline_expr_position_modules_expr(
                item,
                caller=caller,
                inline_modules=inline_modules,
                blocked_names=blocked_names,
                need_atomic=True,
                protected_names=protected_names,
            )
            prefix.extend(item_prefix)
            rewritten_items.append(item_expr)
        return tuple(prefix), _atomicize(replace(expr, items=tuple(rewritten_items)))
    return tuple(), expr


def _inline_expr_position_modules_stmts(
    statements: tuple[AxonStatement, ...],
    *,
    caller: AxonDefinition,
    inline_modules: Mapping[str, AxonDefinition],
    protected_names: set[str] | None = None,
) -> tuple[AxonStatement, ...]:
    blocked = set(_param_names(caller))
    blocked.update(_bound_names_statements(statements))
    rewritten: list[AxonStatement] = []
    for stmt in statements:
        if isinstance(stmt, AxonBind):
            prefix, expr = _inline_expr_position_modules_expr(
                stmt.expr,
                caller=caller,
                inline_modules=inline_modules,
                blocked_names=blocked,
                need_atomic=False,
                protected_names=protected_names,
            )
            rewritten.extend(prefix)
            rewritten.append(replace(stmt, expr=expr))
            continue
        if isinstance(stmt, AxonReturn | AxonYield):
            new_values: list[AxonExpr] = []
            for value in stmt.values:
                prefix, rewritten_value = _inline_expr_position_modules_expr(
                    value,
                    caller=caller,
                    inline_modules=inline_modules,
                    blocked_names=blocked,
                    need_atomic=True,
                    protected_names=protected_names,
                )
                rewritten.extend(prefix)
                new_values.append(rewritten_value)
            rewritten.append(replace(stmt, values=tuple(new_values)))
            continue
        rewritten.append(stmt)
    return tuple(rewritten)


def _inline_expression_position_modules(program: AxonFile, *, main_module: str | None) -> AxonFile:
    callsites = _module_callsites(program)
    protected_names: set[str] = set()
    inline_modules: dict[str, AxonDefinition] = {}
    for module in program.modules:
        callers = callsites.get(module.name, [])
        if main_module is not None and module.name == main_module:
            continue
        if len(callers) != 1:
            continue
        if not _is_straight_line_module(module):
            continue
        inline_modules[module.name] = module
    if not inline_modules:
        return program
    rewritten = replace(
        program,
        modules=tuple(
            replace(
                module,
                statements=_inline_expr_position_modules_stmts(
                    module.statements,
                    caller=module,
                    inline_modules=inline_modules,
                    protected_names=protected_names,
                ),
            )
            for module in program.modules
        ),
    )
    return _prune_unreachable_modules(rewritten, main_module=main_module)


def _inline_single_callsite_modules(program: AxonFile, *, main_module: str | None) -> AxonFile:
    with _opt_debug_time("inline_single_callsite_modules"):
        callsites = _module_callsites(program)
        protected_names: set[str] = set()

        inline_modules: dict[str, AxonDefinition] = {}
        for module in program.modules:
            callers = callsites.get(module.name, [])
            if main_module is not None and module.name == main_module:
                continue
            if len(callers) != 1:
                continue
            if callers and callers[0][0] == module.name:
                continue
            if not _is_straight_line_module(module):
                continue
            inline_modules[module.name] = module

        if not inline_modules:
            return program

        rewritten_modules: list[AxonDefinition] = []
        for module in program.modules:
            rewritten_modules.append(
                replace(
                    module,
                    statements=_inline_call_bind_statements(
                        module.statements,
                        caller=module,
                        inline_modules=inline_modules,
                        protected_names=protected_names,
                    ),
                )
            )
        rewritten = replace(program, modules=tuple(rewritten_modules))
        return _prune_unreachable_modules(rewritten, main_module=main_module)


def _resolve_constant_actual(
    expr: AxonExpr,
    known: Mapping[str, AxonExpr],
    *,
    caller_dynamic_names: set[str],
    constant_names: set[str],
) -> AxonExpr | None:
    if isinstance(expr, AxonExprAscribe):
        resolved = _resolve_constant_actual(
            expr.expr,
            known,
            caller_dynamic_names=caller_dynamic_names,
            constant_names=constant_names,
        )
        if resolved is not None and _constant_actual_key(resolved) == _constant_actual_key(
            _unwrap_expr(expr)
        ):
            return expr
        return resolved
    if isinstance(expr, AxonExprParen):
        resolved = _resolve_constant_actual(
            expr.inner,
            known,
            caller_dynamic_names=caller_dynamic_names,
            constant_names=constant_names,
        )
        if resolved is not None and _constant_actual_key(resolved) == _constant_actual_key(
            _unwrap_expr(expr)
        ):
            return expr
        return resolved
    if _is_atomic_expr(expr):
        if isinstance(expr, AxonExprName):
            resolved = known.get(expr.name)
            if resolved is not None:
                return resolved
            if expr.name not in caller_dynamic_names:
                return expr if expr.name in constant_names else None
            return None
        if isinstance(expr, AxonExprPath):
            return None
        return expr
    return None


def _join_constant_actual(
    current: AxonExpr | None, candidate: AxonExpr | None
) -> AxonExpr | None | bool:
    if candidate is None:
        return current
    if current is None:
        return candidate
    if _constant_actual_key(current) == _constant_actual_key(candidate):
        return current
    return False


def _constant_actual_key(expr: AxonExpr) -> tuple[object, ...]:
    if isinstance(expr, AxonExprAscribe):
        return ("ascribe", _constant_actual_key(expr.expr), expr.type_expr)
    if isinstance(expr, AxonExprName):
        return ("name", expr.name)
    if isinstance(expr, AxonExprInt):
        return ("int", expr.value)
    if isinstance(expr, AxonExprFloat):
        return ("float", expr.value, expr.lexeme)
    if isinstance(expr, AxonExprBool):
        return ("bool", expr.value)
    if isinstance(expr, AxonExprNull):
        return ("null",)
    if isinstance(expr, AxonExprString):
        return ("string", expr.value)
    if isinstance(expr, AxonExprPath):
        return ("path", expr.absolute, expr.parts)
    return ("other", repr(expr))


def _specialize_modules_by_constant_params(program: AxonFile) -> AxonFile:
    with _opt_debug_time("specialize_modules_by_constant_params"):
        graph = _module_call_graph(program)
        callsites = _module_callsites(program)
        sccs = _module_sccs(graph)
        modules_by_name = {module.name: module for module in program.modules}
        param_names_by_module = {
            module.name: _param_names(module) for module in program.modules
        }
        param_name_sets_by_module = {
            name: set(param_names) for name, param_names in param_names_by_module.items()
        }
        bound_names_by_module = {
            module.name: _bound_names_statements(module.statements)
            for module in program.modules
        }
        call_actuals_by_module: dict[str, list[tuple[str, dict[str, AxonExpr]]]] = {
            module.name: [] for module in program.modules
        }
        constant_names: set[str] = set()
        for module_name, module_callsites in callsites.items():
            module = modules_by_name[module_name]
            call_actuals_by_module[module_name] = [
                (caller_name, _call_actual_by_param(module, call))
                for caller_name, call in module_callsites
            ]
        known_by_module: dict[str, dict[str, AxonExpr]] = {
            module.name: {} for module in program.modules
        }
        for component in sccs:
            component_set = set(component)
            recursive_component = len(component) > 1 or any(
                module_name in graph.get(module_name, set()) for module_name in component
            )
            changed = True
            iteration = 0
            while changed:
                iteration += 1
                if _OPT_DEBUG_ACTIVE and iteration > 1000:
                    snapshot = {
                        name: tuple(sorted(known_by_module[name]))
                        for name in component
                    }
                    raise RuntimeError(
                        "Axon optimize constant specialization did not converge "
                        f"for component {component!r}; known={snapshot!r}"
                    )
                changed = False
                for module_name in component:
                    param_names = param_names_by_module[module_name]
                    candidates: dict[str, AxonExpr] = dict(known_by_module[module_name])
                    valid = {name: True for name in param_names}
                    for caller_name, actuals in call_actuals_by_module[module_name]:
                        if caller_name in component_set:
                            caller_known = known_by_module.get(caller_name, {})
                            caller_bound_names = bound_names_by_module.get(caller_name, set())
                            for name in param_names:
                                if name not in actuals or not valid.get(name, True):
                                    continue
                                actual_inner = _unwrap_expr(actuals[name])
                                if isinstance(actual_inner, AxonExprName):
                                    if actual_inner.name in caller_bound_names:
                                        valid[name] = False
                                        candidates.pop(name, None)
                                        continue
                                    resolved = caller_known.get(actual_inner.name)
                                    if resolved is not None:
                                        joined = _join_constant_actual(
                                            candidates.get(name), resolved
                                        )
                                        if joined is False:
                                            valid[name] = False
                                            candidates.pop(name, None)
                                            continue
                                        if (
                                            isinstance(joined, AxonExpr)
                                            and not (
                                                isinstance(joined, AxonExprName)
                                                and joined.name == name
                                            )
                                            and name not in candidates
                                        ):
                                            candidates[name] = joined
                                    elif actual_inner.name != name:
                                        valid[name] = False
                                        candidates.pop(name, None)
                                    continue
                                valid[name] = False
                                candidates.pop(name, None)
                            continue
                        caller_known = known_by_module.get(caller_name, {})
                        caller_param_names = param_name_sets_by_module.get(caller_name, set())
                        caller_bound_names = bound_names_by_module.get(caller_name, set())
                        caller_dynamic_names = caller_param_names | caller_bound_names
                        for name in param_names:
                            if name not in actuals or not valid.get(name, True):
                                continue
                            actual_expr = actuals[name]
                            resolved = _resolve_constant_actual(
                                actual_expr,
                                caller_known,
                                caller_dynamic_names=caller_dynamic_names,
                                constant_names=constant_names,
                            )
                            if resolved is None:
                                valid[name] = False
                                candidates.pop(name, None)
                                continue
                            joined = _join_constant_actual(candidates.get(name), resolved)
                            if joined is False:
                                valid[name] = False
                                candidates.pop(name, None)
                                continue
                            if isinstance(joined, AxonExprName) and joined.name == name:
                                continue
                            if isinstance(joined, AxonExpr) and name not in candidates:
                                candidates[name] = joined
                    for name in list(candidates):
                        if not valid.get(name, True):
                            candidates.pop(name, None)
                    if candidates != known_by_module[module_name]:
                        known_by_module[module_name] = candidates
                        changed = True
                if not recursive_component:
                    break
        rewritten_modules: list[AxonDefinition] = []
        specs: dict[str, tuple[AxonDefinition, AxonDefinition]] = {}
        for module in program.modules:
            subst = {
                name: expr
                for name, expr in known_by_module[module.name].items()
                if not (isinstance(expr, AxonExprName) and expr.name == name)
            }
            subst = _filter_dim_safe_param_subst(module, subst)
            if not subst:
                specs[module.name] = (module, module)
                rewritten_modules.append(module)
                continue
            new_module = replace(
                module,
                params=tuple(param for param in module.params if param.name not in subst),
                statements=_substitute_stmts(module.statements, subst),
            )
            specs[module.name] = (module, new_module)
            rewritten_modules.append(new_module)
        rewritten_program = replace(program, modules=tuple(rewritten_modules))
        final_modules = tuple(
            replace(module, statements=_rewrite_calls_stmts(module.statements, specs))
            for module in rewritten_program.modules
        )
        return replace(
            rewritten_program,
            modules=final_modules,
        )


def _reachable_modules(program: AxonFile, *, root: str | None) -> frozenset[str]:
    if root is None:
        return frozenset(module.name for module in program.modules)
    graph = _module_call_graph(program)
    seen: set[str] = set()
    stack = [root]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(sorted(graph.get(current, ())))
    return frozenset(seen)


def _prune_unreachable_modules(program: AxonFile, *, main_module: str | None) -> AxonFile:
    with _opt_debug_time("prune_unreachable_modules"):
        keep = _reachable_modules(program, root=main_module)
        return replace(
            program, modules=tuple(module for module in program.modules if module.name in keep)
        )


def optimize_flat_typed_axon_file(program: AxonFile, *, main_module: str | None = None) -> AxonFile:
    program = _prune_unreachable_modules(program, main_module=main_module)
    validate_typed_axon_file(program, main_module=main_module)
    current = program
    pass_index = 0
    while True:
        pass_index += 1
        current_before_pass = current
        _opt_debug_begin_pass(pass_index=pass_index)
        with _opt_debug_time("pass_constraint_fold"):
            constraint_folded_modules = tuple(
                replace(
                    module,
                    statements=_fold_statements_by_known_bool(
                        module.statements,
                        _known_bool_constraints(module),
                        _known_literal_constraints(module),
                    ),
                )
                for module in current.modules
            )
        current = replace(current, modules=constraint_folded_modules)
        specialized = _specialize_modules_by_constant_params(current)
        try:
            validate_flat_axon_file(specialized, main_module=main_module)
        except ValueError:
            specialized = current
        single_callsite_specialized = _specialize_single_callsite_modules(
            specialized, main_module=main_module
        )
        try:
            validate_flat_axon_file(single_callsite_specialized, main_module=main_module)
            specialized = single_callsite_specialized
        except ValueError:
            pass
        inlined = _inline_single_callsite_modules(specialized, main_module=main_module)
        try:
            validate_flat_axon_file(inlined, main_module=main_module)
            specialized = inlined
        except ValueError:
            pass
        alias_modules = {
            module.name: (module, alias_expr)
            for module in specialized.modules
            if (alias_expr := _module_alias_expr(module)) is not None
        }
        with _opt_debug_time("pass_optimize_statements"):
            optimized_modules = tuple(
                replace(
                    module,
                    statements=_optimize_statements_fixpoint(
                        module.statements, alias_modules=alias_modules
                    ),
                )
                for module in specialized.modules
            )
        optimized = replace(specialized, modules=optimized_modules)
        try:
            validate_flat_axon_file(optimized, main_module=main_module)
        except ValueError:
            optimized = specialized
        optimized = _prune_unused_module_params(optimized)
        validate_flat_axon_file(optimized, main_module=main_module)
        optimized = _prune_unreachable_modules(optimized, main_module=main_module)
        optimized = replace(
            optimized,
            modules=tuple(
                replace(module, statements=_atomicize_call_args_statements(module.statements))
                for module in optimized.modules
            ),
        )
        _opt_debug_end_pass(modules=len(optimized.modules))
        with _opt_debug_time("pass_retype"):
            retyped = typecheck2_flat_axon_file(optimized, main_module=main_module)
        with _opt_debug_time("pass_validate_typed"):
            validate_typed_axon_file(retyped, main_module=main_module)
        if ast_equal(current, retyped):
            _opt_debug_print_diff(before=current_before_pass, after=retyped, pass_index=pass_index)
            validate_optimized_flat_typed_axon_file(retyped, main_module=main_module)
            return retyped
        _opt_debug_print_diff(before=current_before_pass, after=retyped, pass_index=pass_index)
        current = retyped


def optimize_safe_flat_typed_axon_file(
    program: AxonFile,
    *,
    main_module: str | None = None,
    max_iterations: int = 64,
) -> AxonFile:
    """Run only conservative pre-Graph-IR optimizations on flat typed Axon."""

    selected_main = resolve_main_module(program, main_module=main_module)
    validate_flat_axon_file(program, main_module=selected_main)
    validate_typed_axon_file(program, main_module=selected_main)
    current = prune_unreachable_definitions(program, entrypoint=selected_main)
    validate_typed_axon_file(current, main_module=selected_main)
    for _ in range(max_iterations):
        rewritten = replace(
            current,
            modules=tuple(
                _optimize_safe_module(module)
                for module in current.modules
            ),
        )
        validate_flat_axon_file(rewritten, main_module=selected_main)
        retyped = typecheck2_flat_axon_file(rewritten, main_module=selected_main)
        retyped = prune_unreachable_definitions(retyped, entrypoint=selected_main)
        validate_typed_axon_file(retyped, main_module=selected_main)
        if ast_equal(current, retyped):
            return retyped
        current = retyped
    raise RuntimeError(
        "safe Axon optimization failed to converge within "
        f"{max_iterations} iterations for main module {selected_main!r}"
    )


def normalize_backend_required_flat_typed_axon_file(
    program: AxonFile, *, main_module: str | None = None
) -> AxonFile:
    """Run flat-Axon rewrites required by the current backend contract.

    These are not semantic optimizations and therefore still run when CLI
    optimization is disabled.
    """

    validate_typed_axon_file(program, main_module=main_module)
    normalized = replace(
        program,
        modules=tuple(
            replace(module, statements=_normalize_list_destructure_binds(module.statements))
            for module in program.modules
        ),
    )
    normalized = typecheck2_flat_axon_file(normalized, main_module=main_module)
    validate_backend_required_flat_typed_axon_file(normalized, main_module=main_module)
    return normalized


__all__ = [
    "normalize_backend_required_flat_typed_axon_file",
    "optimize_flat_typed_axon_file",
    "optimize_safe_flat_typed_axon_file",
]
