from __future__ import annotations

import re
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Any

from ...ops import get_op_lowering_type_signature, get_op_type_rule
from ...ops._broadcast import broadcast_shape
from ..ast import (
    AxonBind,
    AxonCond,
    AxonDefinition,
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
    AxonParam,
    AxonRepeat,
    AxonReturn,
    AxonScopeBind,
    AxonStatement,
    AxonYield,
    DimExprBinary,
    DimToken,
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
    dim_token_names,
)
from ..typecheck.core import (
    _PrimitiveTypeHelpers,
    _TcCtx,
    _annotate_expr,
    _apply_subst,
    _bind_dim_name,
    _broadcast_tensor_branch_types,
    _destructure_type,
    _expr_to_dim_token,
    _expr_to_dim_token_resolved,
    _expand_alias,
    _is_type_expr_instance,
    _is_scalar_numeric_type,
    _join_branch_types,
    _normalize_dim_token,
    _normalize_expr,
    _normalize_statement,
    _normalize_type_expr_for_module,
    _primitive_op_name,
    _resolve_type_dim_aliases,
    _scoped_typevars,
    _type_dims,
    _type_expr_from_spec,
    _unify,
    _unify_broadcast_tensor_dims,
)
from ..entrypoint import resolve_main_module
from ..resolve import reachable_definitions
from ..validate import validate_flat_axon_file, validate_typed_axon_file

_PATH_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _call_base_and_surface(callee: str) -> tuple[str, str]:
    indexes = [idx for idx in (callee.find("@"), callee.find("::")) if idx >= 0]
    if not indexes:
        return callee, ""
    idx = min(indexes)
    return callee[:idx], callee[idx:]


def _add_path_placeholder_refs(text: str, out: set[str], names: set[str]) -> None:
    for match in _PATH_PLACEHOLDER_RE.finditer(text):
        name = match.group(1)
        if name in names:
            out.add(name)


@dataclass
class _CallBinding:
    args: tuple[AxonExpr, ...]
    kwargs: dict[str, AxonKwargValue]
    param_types: dict[str, TypeExpr]
    path_types: dict[str, TypeExpr]
    return_types: tuple[TypeExpr, ...] | None
    expr_defs: dict[str, AxonExpr]
    dim_bindings: dict[str, DimToken]


@dataclass
class _Tc2:
    program: AxonFile
    main_module: str | None
    modules_by_name: dict[str, AxonDefinition]
    ctx: _TcCtx
    typed_modules: dict[str, AxonDefinition]
    specialization_conflicts: set[str]
    fresh_dim_names: set[str]
    fresh_dim_sources: dict[str, str]
    fresh_type_names: set[str]
    fresh_type_sources: dict[str, str]
    active: tuple[str, ...] = ()


def _unify_tc2(left: TypeExpr, right: TypeExpr, ctx: _TcCtx) -> TypeExpr:
    left_expanded = _expand_alias(_apply_subst(left, ctx), ctx)
    right_expanded = _expand_alias(_apply_subst(right, ctx), ctx)
    if isinstance(left_expanded, TypeTensor) and isinstance(right_expanded, TypeTensor):
        if (
            len(left_expanded.dims) == len(right_expanded.dims)
            and not any(
                isinstance(dim, str) and dim.startswith("..")
                for dim in (*left_expanded.dims, *right_expanded.dims)
            )
            and all(
                _dim_equivalent(left_dim, right_dim, ctx)
                for left_dim, right_dim in zip(left_expanded.dims, right_expanded.dims, strict=True)
            )
        ):
            return TypeTensor(base=right_expanded.base, dims=right_expanded.dims)
    return _unify(left_expanded, right_expanded, ctx)


def _module_graph(program: AxonFile) -> dict[str, set[str]]:
    names = {module.name for module in program.modules}
    graph = {module.name: set[str]() for module in program.modules}

    def visit_expr(expr: AxonExpr, out: set[str]) -> None:
        if isinstance(expr, AxonExprName):
            base, surface = _call_base_and_surface(expr.name)
            if base in names:
                out.add(base)
            _add_path_placeholder_refs(surface, out, names)
            return
        if isinstance(expr, AxonExprPath):
            for part in expr.parts:
                _add_path_placeholder_refs(part, out, names)
            return
        if isinstance(expr, AxonExprCall):
            base, surface = _call_base_and_surface(expr.callee)
            if base in names:
                out.add(base)
            _add_path_placeholder_refs(surface, out, names)
            for arg in expr.args:
                visit_expr(arg, out)
            for value in expr.kwargs.values():
                if isinstance(value, AxonExpr):
                    visit_expr(value, out)
            return
        if isinstance(expr, AxonExprBinary):
            visit_expr(expr.left, out)
            visit_expr(expr.right, out)
            return
        if isinstance(expr, AxonExprBind):
            visit_expr(expr.value, out)
            visit_expr(expr.body, out)
            return
        if isinstance(expr, AxonExprIf | AxonExprTernary):
            visit_expr(expr.cond, out)
            visit_expr(expr.true_expr, out)
            visit_expr(expr.false_expr, out)
            return
        if isinstance(expr, AxonExprLambda):
            visit_expr(expr.body, out)
            return
        if isinstance(expr, AxonExprAscribe):
            visit_expr(expr.expr, out)
            return
        if isinstance(expr, AxonExprList | AxonExprTuple):
            for item in expr.items:
                visit_expr(item, out)
            return
        if isinstance(expr, AxonExprParen):
            visit_expr(expr.inner, out)
            return
        if isinstance(expr, AxonExprPipe):
            visit_expr(expr.value, out)
            for stage in expr.stages:
                visit_expr(stage, out)
            return
        if isinstance(expr, AxonExprDo):
            for stmt in expr.body:
                visit_stmt(stmt, out)

    def visit_stmt(stmt: AxonStatement, out: set[str]) -> None:
        if isinstance(stmt, AxonBind):
            visit_expr(stmt.expr, out)
            return
        if isinstance(stmt, AxonReturn | AxonYield):
            for value in stmt.values:
                visit_expr(value, out)
            return
        if isinstance(stmt, AxonCond):
            visit_expr(stmt.cond, out)
            for item in stmt.true_body:
                visit_stmt(item, out)
            for item in stmt.false_body:
                visit_stmt(item, out)
            return
        if isinstance(stmt, AxonRepeat):
            visit_expr(stmt.from_expr, out)
            visit_expr(stmt.to_expr, out)
            visit_expr(stmt.step_expr, out)
            for item in stmt.body:
                visit_stmt(item, out)
            return
        if isinstance(stmt, AxonScopeBind):
            for kw_value in stmt.kwargs.values():
                if not isinstance(kw_value, bool | int | float | str | list) and kw_value is not None:
                    visit_expr(kw_value, out)
            for item in stmt.body:
                visit_stmt(item, out)

    for module in program.modules:
        for statement in module.statements:
            visit_stmt(statement, graph[module.name])
        if module.body_expr is not None:
            visit_expr(module.body_expr, graph[module.name])
    return graph


def _reachable(program: AxonFile, main_module: str | None) -> set[str]:
    main_module = resolve_main_module(program, main_module=main_module)
    module_names = {module.name for module in program.modules}
    if main_module not in module_names:
        raise ValueError(f"Axon typecheck2 failed: main definition {main_module!r} not found")
    return set(reachable_definitions(program, entrypoint=main_module))


def _prune_to_main(program: AxonFile, main_module: str | None) -> AxonFile:
    keep = _reachable(program, main_module)
    if len(keep) == len(program.modules):
        return program
    return replace(program, modules=tuple(module for module in program.modules if module.name in keep))


def _annotate(ctx: _TcCtx, expr: AxonExpr, tp: TypeExpr, *, arity: int = 1) -> AxonExpr:
    return _annotate_expr(expr, tp, arity=arity, ctx=ctx)


def _collect_dim_names(tp: TypeExpr | None) -> set[str]:
    if tp is None:
        return set()
    if isinstance(tp, TypeOptional):
        return _collect_dim_names(tp.inner)
    if isinstance(tp, TypeList):
        return _collect_dim_names(tp.item)
    if isinstance(tp, TypeTuple):
        out: set[str] = set()
        for item in tp.items:
            out.update(_collect_dim_names(item))
        return out
    if isinstance(tp, TypeTensor):
        names: set[str] = set()
        for dim in tp.dims:
            names.update(dim_token_names(dim))
        return names
    return set()


def _module_header_env(module: AxonDefinition, ctx: _TcCtx) -> dict[str, TypeExpr]:
    env: dict[str, TypeExpr] = {}
    for name in module.path_params:
        env[name] = TypePath()
    if module.path_param is not None:
        env[module.path_param] = TypePath()
    for param in module.params:
        tp = _scoped_typevars(
            param.type_expr,
            module_name=module.name,
            ctx=ctx,
            freshen_generics=False,
        )
        has_non_null_default = param.default_expr is not None and not isinstance(
            param.default_expr, AxonExprNull
        )
        env[param.name] = (
            TypeOptional(tp)
            if param.optional and not has_non_null_default and not isinstance(tp, TypeOptional)
            else tp
        )
    for param in module.params:
        for dim in _collect_dim_names(param.type_expr):
            env.setdefault(dim, TypeDim())
    for dim in _collect_dim_names(module.return_type_expr):
        env.setdefault(dim, TypeDim())
    return env


def _return_types(module: AxonDefinition, ctx: _TcCtx) -> tuple[TypeExpr, ...] | None:
    if module.return_type_expr is None:
        return None
    tp = _scoped_typevars(
        module.return_type_expr,
        module_name=module.name,
        ctx=ctx,
        freshen_generics=False,
    )
    if isinstance(tp, TypeTuple):
        return tp.items
    return (tp,)


def _is_generic_named_type(tp: TypeExpr, ctx: _TcCtx) -> bool:
    return (
        isinstance(tp, TypeNamed)
        and tp.name != "Tensor"
        and tp.name not in ctx.type_aliases
        and "." not in tp.name
        and "::" not in tp.name
    )


def _instantiate_call_signature(
    module: AxonDefinition, state: _Tc2
) -> tuple[list[TypeExpr], tuple[TypeExpr, ...] | None]:
    ctx = state.ctx
    dim_subst: dict[str, DimToken] = {}
    type_subst: dict[str, TypeVar] = {}

    def fresh_dim(name: str) -> str:
        existing = dim_subst.get(name)
        if isinstance(existing, str):
            return existing
        ctx.fresh_counter += 1
        fresh = f"__d{ctx.fresh_counter}"
        if name.startswith(".."):
            fresh = ".." + fresh
        dim_subst[name] = fresh
        state.fresh_dim_names.add(fresh)
        state.fresh_dim_sources[fresh] = name
        return fresh

    def rewrite_dim(dim: DimToken) -> DimToken:
        if isinstance(dim, str):
            return fresh_dim(dim)
        if isinstance(dim, int):
            return dim
        return DimExprBinary(op=dim.op, left=rewrite_dim(dim.left), right=rewrite_dim(dim.right))

    def fresh_type(name: str) -> TypeVar:
        existing = type_subst.get(name)
        if existing is not None:
            return existing
        fresh = ctx.fresh_type_var()
        type_subst[name] = fresh
        state.fresh_type_names.add(fresh.name)
        state.fresh_type_sources[fresh.name] = name
        return fresh

    def rewrite_type(tp: TypeExpr | None) -> TypeExpr:
        if tp is None:
            return TypeAny()
        if isinstance(tp, TypeVar):
            return fresh_type(tp.name)
        if _is_generic_named_type(tp, ctx):
            assert isinstance(tp, TypeNamed)
            return fresh_type(tp.name)
        if isinstance(tp, TypeOptional):
            return TypeOptional(inner=rewrite_type(tp.inner))
        if isinstance(tp, TypeList):
            return TypeList(item=rewrite_type(tp.item))
        if isinstance(tp, TypeTuple):
            return TypeTuple(items=tuple(rewrite_type(item) for item in tp.items))
        if isinstance(tp, TypeNamed):
            return TypeNamed(name=tp.name, args=tuple(rewrite_dim(dim) for dim in tp.args))
        if isinstance(tp, TypeTensor):
            return TypeTensor(base=tp.base, dims=tuple(rewrite_dim(dim) for dim in tp.dims))
        return tp

    param_types: list[TypeExpr] = []
    for param in module.params:
        tp = rewrite_type(param.type_expr)
        param_types.append(TypeOptional(tp) if param.optional and not isinstance(tp, TypeOptional) else tp)
    return_tp = rewrite_type(module.return_type_expr) if module.return_type_expr is not None else None
    if return_tp is None:
        return param_types, None
    if isinstance(return_tp, TypeTuple):
        return param_types, return_tp.items
    return param_types, (return_tp,)


def _target_type(tp: TypeExpr, arity: int, ctx: _TcCtx) -> tuple[TypeExpr, ...]:
    return _destructure_type(tp, arity, ctx)


def _merge_return_types(
    current: tuple[TypeExpr, ...] | None, returned: tuple[TypeExpr, ...], ctx: _TcCtx
) -> tuple[TypeExpr, ...]:
    if current is None:
        return tuple(_apply_subst(tp, ctx) for tp in returned)
    if len(current) != len(returned):
        raise ValueError("Axon typecheck2 failed: inconsistent return arity")
    return tuple(_join_branch_types(a, b, ctx) for a, b in zip(current, returned, strict=True))


def _binary_arithmetic_type(left: TypeExpr, right: TypeExpr, ctx: _TcCtx) -> TypeExpr:
    left_expanded = _expand_alias(_apply_subst(left, ctx), ctx)
    right_expanded = _expand_alias(_apply_subst(right, ctx), ctx)
    if isinstance(left_expanded, TypeTensor) and isinstance(right_expanded, TypeTensor):
        if any(isinstance(dim, str) and dim.startswith("..") for dim in left_expanded.dims):
            return _apply_subst(left, ctx)
        if any(isinstance(dim, str) and dim.startswith("..") for dim in right_expanded.dims):
            return _apply_subst(left, ctx)
        return _broadcast_tensor_branch_types(left, right, ctx)
    if isinstance(left_expanded, TypeTensor) and _is_scalar_numeric_type(right_expanded):
        return _apply_subst(left, ctx)
    if isinstance(right_expanded, TypeTensor) and _is_scalar_numeric_type(left_expanded):
        return _apply_subst(right, ctx)
    if isinstance(left_expanded, TypeTensor):
        return _apply_subst(left, ctx)
    if isinstance(right_expanded, TypeTensor):
        return _apply_subst(right, ctx)
    if isinstance(left_expanded, TypeVar | TypeAny) and _is_scalar_numeric_type(right_expanded):
        return _apply_subst(left, ctx)
    if isinstance(right_expanded, TypeVar | TypeAny) and _is_scalar_numeric_type(left_expanded):
        return _apply_subst(right, ctx)
    if isinstance(left_expanded, TypeDim) or isinstance(right_expanded, TypeDim):
        _unify(left, TypeDim(), ctx)
        _unify(right, TypeDim(), ctx)
        return TypeDim()
    if isinstance(left_expanded, TypeVar | TypeAny) and isinstance(right_expanded, TypeFloat | TypeInt):
        return _apply_subst(left, ctx)
    if isinstance(right_expanded, TypeVar | TypeAny) and isinstance(left_expanded, TypeFloat | TypeInt):
        return _apply_subst(right, ctx)
    if isinstance(left_expanded, TypeVar | TypeAny) and isinstance(right_expanded, TypeVar | TypeAny):
        return _apply_subst(left, ctx)
    if isinstance(left_expanded, TypeFloat) or isinstance(right_expanded, TypeFloat):
        _unify(left, TypeFloat(), ctx)
        _unify(right, TypeFloat(), ctx)
        return TypeFloat()
    if isinstance(left_expanded, TypeInt) and isinstance(right_expanded, TypeInt):
        _unify(left, TypeInt(), ctx)
        _unify(right, TypeInt(), ctx)
        return TypeInt()
    fallback = ctx.fresh_type_var()
    _unify(left, fallback, ctx)
    _unify(right, fallback, ctx)
    return fallback


def _final_return_type(types: tuple[TypeExpr, ...] | None, ctx: _TcCtx) -> TypeExpr | None:
    if types is None:
        return None
    if len(types) == 1:
        return _apply_subst(types[0], ctx)
    return TypeTuple(items=tuple(_apply_subst(tp, ctx) for tp in types))


def _join_expr_branch_types(left: TypeExpr, right: TypeExpr, ctx: _TcCtx) -> TypeExpr:
    try:
        return _join_branch_types(left, right, ctx)
    except ValueError:
        left_expanded = _expand_alias(_apply_subst(left, ctx), ctx)
        right_expanded = _expand_alias(_apply_subst(right, ctx), ctx)
        if (
            isinstance(left_expanded, TypeTensor)
            and isinstance(right_expanded, TypeTensor)
            and len(left_expanded.dims) == len(right_expanded.dims)
        ):
            dims: list[DimToken] = []
            for left_dim, right_dim in zip(left_expanded.dims, right_expanded.dims, strict=True):
                if left_dim == right_dim:
                    dims.append(left_dim)
                elif _dim_equivalent(left_dim, right_dim, ctx):
                    dims.append(left_dim)
                else:
                    raise
            return TypeTensor(base=left_expanded.base, dims=tuple(dims))
        raise


def _is_fresh_dim_name(name: str, state: _Tc2) -> bool:
    return name in state.fresh_dim_names


def _is_generated_definition_name(name: str) -> bool:
    return "__" in name


def _is_loop_generated_definition_name(name: str) -> bool:
    return "__loop_" in name and ("_recur_" in name or "_recur_continue_" in name)


def _definition_signature(module: AxonDefinition) -> tuple[tuple[TypeExpr | None, ...], TypeExpr | None]:
    return tuple(param.type_expr for param in module.params), module.return_type_expr


def _dim_token_to_expr(dim: DimToken) -> AxonExpr | None:
    if isinstance(dim, int):
        return AxonExprInt(value=dim)
    if isinstance(dim, str) and not dim.startswith(".."):
        return AxonExprName(name=dim)
    if isinstance(dim, DimExprBinary):
        left = _dim_token_to_expr(dim.left)
        right = _dim_token_to_expr(dim.right)
        if left is not None and right is not None:
            if isinstance(left, AxonExprBinary):
                left = AxonExprParen(inner=left)
            if isinstance(right, AxonExprBinary):
                right = AxonExprParen(inner=right)
            return AxonExprBinary(left=left, op=dim.op, right=right)
    return None


def _collect_dim_replacements_from_types(
    original: TypeExpr | None,
    specialized: TypeExpr | None,
    out: dict[str, AxonExpr],
) -> None:
    if original is None or specialized is None:
        return
    if isinstance(original, TypeOptional) and isinstance(specialized, TypeOptional):
        _collect_dim_replacements_from_types(original.inner, specialized.inner, out)
        return
    if isinstance(original, TypeList) and isinstance(specialized, TypeList):
        _collect_dim_replacements_from_types(original.item, specialized.item, out)
        return
    if isinstance(original, TypeTuple) and isinstance(specialized, TypeTuple):
        for original_item, specialized_item in zip(
            original.items, specialized.items, strict=False
        ):
            _collect_dim_replacements_from_types(original_item, specialized_item, out)
        return
    if isinstance(original, TypeTensor) and isinstance(specialized, TypeTensor):
        for original_dim, specialized_dim in zip(
            original.dims, specialized.dims, strict=False
        ):
            if (
                isinstance(original_dim, str)
                and not original_dim.startswith("..")
                and original_dim != specialized_dim
            ):
                expr = _dim_token_to_expr(specialized_dim)
                if expr is not None:
                    out[original_dim] = expr
        return
    if isinstance(original, TypeNamed) and isinstance(specialized, TypeNamed):
        for original_dim, specialized_dim in zip(original.args, specialized.args, strict=False):
            if (
                isinstance(original_dim, str)
                and not original_dim.startswith("..")
                and original_dim != specialized_dim
            ):
                expr = _dim_token_to_expr(specialized_dim)
                if expr is not None:
                    out[original_dim] = expr


def _preserve_compound_dim_binders(
    original: TypeExpr | None,
    specialized: TypeExpr | None,
) -> TypeExpr | None:
    if original is None or specialized is None:
        return specialized
    if isinstance(original, TypeOptional) and isinstance(specialized, TypeOptional):
        return replace(
            specialized,
            inner=_preserve_compound_dim_binders(original.inner, specialized.inner)
            or specialized.inner,
        )
    if isinstance(original, TypeList) and isinstance(specialized, TypeList):
        return replace(
            specialized,
            item=_preserve_compound_dim_binders(original.item, specialized.item)
            or specialized.item,
        )
    if isinstance(original, TypeTuple) and isinstance(specialized, TypeTuple):
        return replace(
            specialized,
            items=tuple(
                _preserve_compound_dim_binders(original_item, specialized_item)
                or specialized_item
                for original_item, specialized_item in zip(
                    original.items, specialized.items, strict=False
                )
            ),
        )
    if isinstance(original, TypeTensor) and isinstance(specialized, TypeTensor):
        dims: list[DimToken] = []
        for original_dim, specialized_dim in zip(
            original.dims, specialized.dims, strict=False
        ):
            if isinstance(original_dim, str) and not original_dim.startswith(".."):
                dims.append(original_dim)
            else:
                dims.append(specialized_dim)
        return replace(specialized, dims=tuple(dims))
    if isinstance(original, TypeNamed) and isinstance(specialized, TypeNamed):
        args: list[DimToken] = []
        for original_dim, specialized_dim in zip(
            original.args, specialized.args, strict=False
        ):
            if isinstance(original_dim, str) and not original_dim.startswith(".."):
                args.append(original_dim)
            else:
                args.append(specialized_dim)
        return replace(specialized, args=tuple(args))
    return specialized


def _is_atomic_expr_for_flat(expr: AxonExpr) -> bool:
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
        return all(_is_atomic_expr_for_flat(item) for item in expr.items)
    if isinstance(expr, AxonExprAscribe):
        return _is_atomic_expr_for_flat(expr.expr)
    return False


def _annotate_dim_expr(expr: AxonExpr, ctx: _TcCtx) -> AxonExpr:
    if isinstance(expr, AxonExprInt):
        return _annotate(ctx, expr, TypeInt())
    if isinstance(expr, AxonExprName):
        return _annotate(ctx, expr, TypeDim())
    if isinstance(expr, AxonExprBinary):
        return _annotate(
            ctx,
            replace(
                expr,
                left=_annotate_dim_expr(expr.left, ctx),
                right=_annotate_dim_expr(expr.right, ctx),
            ),
            TypeDim(),
        )
    if isinstance(expr, AxonExprParen):
        return _annotate(ctx, replace(expr, inner=_annotate_dim_expr(expr.inner, ctx)), TypeDim())
    if isinstance(expr, AxonExprAscribe):
        return _annotate(ctx, replace(expr, expr=_annotate_dim_expr(expr.expr, ctx)), TypeDim())
    return _annotate(ctx, expr, TypeDim())


def _expr_uses_name(expr: AxonExpr, name: str, bound: set[str]) -> bool:
    if isinstance(expr, AxonExprName):
        return expr.name == name and expr.name not in bound
    if isinstance(expr, AxonExprBinary):
        return _expr_uses_name(expr.left, name, bound) or _expr_uses_name(expr.right, name, bound)
    if isinstance(expr, AxonExprBind):
        return _expr_uses_name(expr.value, name, bound) or _expr_uses_name(
            expr.body, name, {*bound, expr.var}
        )
    if isinstance(expr, AxonExprIf | AxonExprTernary):
        return (
            _expr_uses_name(expr.cond, name, bound)
            or _expr_uses_name(expr.true_expr, name, bound)
            or _expr_uses_name(expr.false_expr, name, bound)
        )
    if isinstance(expr, AxonExprLambda):
        return _expr_uses_name(expr.body, name, {*bound, expr.var})
    if isinstance(expr, AxonExprAscribe):
        return _expr_uses_name(expr.expr, name, bound)
    if isinstance(expr, AxonExprList | AxonExprTuple):
        return any(_expr_uses_name(item, name, bound) for item in expr.items)
    if isinstance(expr, AxonExprParen):
        return _expr_uses_name(expr.inner, name, bound)
    if isinstance(expr, AxonExprPipe):
        return _expr_uses_name(expr.value, name, bound) or any(
            _expr_uses_name(stage, name, bound) for stage in expr.stages
        )
    if isinstance(expr, AxonExprCall):
        return any(_expr_uses_name(arg, name, bound) for arg in expr.args) or any(
            _expr_uses_name(value, name, bound)
            for value in expr.kwargs.values()
            if isinstance(value, AxonExpr)
        )
    if isinstance(expr, AxonExprDo):
        return _statements_use_name(expr.body, name, set(bound))
    return False


def _statements_use_name(
    statements: tuple[AxonStatement, ...], name: str, bound: set[str]
) -> bool:
    local = set(bound)
    for stmt in statements:
        if isinstance(stmt, AxonBind):
            if _expr_uses_name(stmt.expr, name, local):
                return True
            local.update(target for target in stmt.targets if target != "_")
            continue
        if isinstance(stmt, AxonReturn | AxonYield):
            if any(_expr_uses_name(value, name, local) for value in stmt.values):
                return True
            continue
        if isinstance(stmt, AxonCond):
            if _expr_uses_name(stmt.cond, name, local):
                return True
            if _statements_use_name(stmt.true_body, name, set(local)):
                return True
            if _statements_use_name(stmt.false_body, name, set(local)):
                return True
            continue
        if isinstance(stmt, AxonRepeat):
            carry = stmt.carry or ()
            targets = stmt.targets or ()
            if (
                _expr_uses_name(stmt.from_expr, name, local)
                or _expr_uses_name(stmt.to_expr, name, local)
                or _expr_uses_name(stmt.step_expr, name, local)
            ):
                return True
            loop_bound = {*local, stmt.var, *(item for item in carry if item != "_")}
            if _statements_use_name(stmt.body, name, loop_bound):
                return True
            local.update(target for target in targets if target != "_")
            continue
        if isinstance(stmt, AxonScopeBind):
            if any(
                _expr_uses_name(value, name, local)
                for value in stmt.kwargs.values()
                if isinstance(value, AxonExpr)
            ):
                return True
            if _statements_use_name(stmt.body, name, set(local)):
                return True
            local.update(target for target in stmt.targets if target != "_")
    return False


def _rewrite_expr_names(
    expr: AxonExpr, replacements: dict[str, AxonExpr], bound: set[str]
) -> AxonExpr:
    if isinstance(expr, AxonExprName) and expr.name in replacements and expr.name not in bound:
        return replacements[expr.name]
    if isinstance(expr, AxonExprBinary):
        return replace(
            expr,
            left=_rewrite_expr_names(expr.left, replacements, bound),
            right=_rewrite_expr_names(expr.right, replacements, bound),
        )
    if isinstance(expr, AxonExprBind):
        value = _rewrite_expr_names(expr.value, replacements, bound)
        return replace(
            expr,
            value=value,
            body=_rewrite_expr_names(expr.body, replacements, {*bound, expr.var}),
        )
    if isinstance(expr, AxonExprIf | AxonExprTernary):
        return replace(
            expr,
            cond=_rewrite_expr_names(expr.cond, replacements, bound),
            true_expr=_rewrite_expr_names(expr.true_expr, replacements, bound),
            false_expr=_rewrite_expr_names(expr.false_expr, replacements, bound),
        )
    if isinstance(expr, AxonExprLambda):
        return replace(
            expr, body=_rewrite_expr_names(expr.body, replacements, {*bound, expr.var})
        )
    if isinstance(expr, AxonExprAscribe):
        return replace(expr, expr=_rewrite_expr_names(expr.expr, replacements, bound))
    if isinstance(expr, AxonExprList):
        return replace(
            expr,
            items=tuple(_rewrite_expr_names(item, replacements, bound) for item in expr.items),
        )
    if isinstance(expr, AxonExprTuple):
        return replace(
            expr,
            items=tuple(_rewrite_expr_names(item, replacements, bound) for item in expr.items),
        )
    if isinstance(expr, AxonExprParen):
        return replace(expr, inner=_rewrite_expr_names(expr.inner, replacements, bound))
    if isinstance(expr, AxonExprPipe):
        return replace(
            expr,
            value=_rewrite_expr_names(expr.value, replacements, bound),
            stages=tuple(_rewrite_expr_names(stage, replacements, bound) for stage in expr.stages),
        )
    if isinstance(expr, AxonExprCall):
        return replace(
            expr,
            args=tuple(_rewrite_expr_names(arg, replacements, bound) for arg in expr.args),
            kwargs={
                key: _rewrite_expr_names(value, replacements, bound)
                if isinstance(value, AxonExpr)
                else value
                for key, value in expr.kwargs.items()
            },
        )
    if isinstance(expr, AxonExprDo):
        return replace(expr, body=_rewrite_statements_names(expr.body, replacements, set(bound)))
    return expr


def _rewrite_statements_names(
    statements: tuple[AxonStatement, ...], replacements: dict[str, AxonExpr], bound: set[str]
) -> tuple[AxonStatement, ...]:
    out: list[AxonStatement] = []
    local = set(bound)
    for stmt in statements:
        if isinstance(stmt, AxonBind):
            out.append(replace(stmt, expr=_rewrite_expr_names(stmt.expr, replacements, local)))
            local.update(name for name in stmt.targets if name != "_")
            continue
        if isinstance(stmt, AxonReturn | AxonYield):
            out.append(
                replace(
                    stmt,
                    values=tuple(
                        _rewrite_expr_names(value, replacements, local) for value in stmt.values
                    ),
                )
            )
            continue
        if isinstance(stmt, AxonCond):
            out.append(
                replace(
                    stmt,
                    cond=_rewrite_expr_names(stmt.cond, replacements, local),
                    true_body=_rewrite_statements_names(stmt.true_body, replacements, set(local)),
                    false_body=_rewrite_statements_names(stmt.false_body, replacements, set(local)),
                )
            )
            continue
        if isinstance(stmt, AxonRepeat):
            carry = stmt.carry or ()
            targets = stmt.targets or ()
            loop_bound = {*local, stmt.var, *(name for name in carry if name != "_")}
            out.append(
                replace(
                    stmt,
                    from_expr=_rewrite_expr_names(stmt.from_expr, replacements, local),
                    to_expr=_rewrite_expr_names(stmt.to_expr, replacements, local),
                    step_expr=_rewrite_expr_names(stmt.step_expr, replacements, local),
                    body=_rewrite_statements_names(stmt.body, replacements, loop_bound),
                )
            )
            local.update(name for name in targets if name != "_")
            continue
        if isinstance(stmt, AxonScopeBind):
            out.append(
                replace(
                    stmt,
                    kwargs={
                        key: _rewrite_expr_names(value, replacements, local)
                        if isinstance(value, AxonExpr)
                        else value
                        for key, value in stmt.kwargs.items()
                    },
                    body=_rewrite_statements_names(stmt.body, replacements, set(local)),
                )
            )
            local.update(name for name in stmt.targets if name != "_")
            continue
        out.append(stmt)
    return tuple(out)


def _specialize_definition_body_dim_names(
    state: _Tc2,
    module: AxonDefinition,
    specialized: AxonDefinition,
    call_expr_defs: dict[str, AxonExpr] | None = None,
) -> AxonDefinition:
    collected: dict[str, AxonExpr] = {}
    for original_param, specialized_param in zip(module.params, specialized.params, strict=True):
        _collect_dim_replacements_from_types(
            original_param.type_expr, specialized_param.type_expr, collected
        )
    _collect_dim_replacements_from_types(
        module.return_type_expr, specialized.return_type_expr, collected
    )
    if not collected:
        return specialized
    replacements: dict[str, AxonExpr] = {}
    prefix_bindings: list[AxonStatement] = []
    source_statements = specialized.statements
    for name, expr in collected.items():
        expr = _annotate_dim_expr(expr, state.ctx)
        if _is_atomic_expr_for_flat(expr):
            replacements[name] = expr
        elif _statements_use_name(source_statements, name, bound=set()):
            prefix_bindings.append(AxonBind(targets=(name,), expr=expr))
    bound = {
        *(param.name for param in specialized.params),
        *specialized.path_params,
    }
    if specialized.path_param is not None:
        bound.add(specialized.path_param)
    if specialized.body_expr is not None:
        return replace(
            specialized,
            body_expr=_rewrite_expr_names(specialized.body_expr, replacements, bound),
        )
    return replace(
        specialized,
        statements=(
            *prefix_bindings,
            *_rewrite_statements_names(specialized.statements, replacements, bound),
        ),
    )


def _type_contains_typevar(tp: TypeExpr | None, ctx: _TcCtx) -> bool:
    if tp is None:
        return False
    if isinstance(tp, TypeVar):
        return True
    if isinstance(tp, TypeNamed) and _is_generic_named_type(tp, ctx):
        return True
    if isinstance(tp, TypeOptional):
        return _type_contains_typevar(tp.inner, ctx)
    if isinstance(tp, TypeList):
        return _type_contains_typevar(tp.item, ctx)
    if isinstance(tp, TypeTuple):
        return any(_type_contains_typevar(item, ctx) for item in tp.items)
    return False


def _definition_signature_contains_typevar(module: AxonDefinition, ctx: _TcCtx) -> bool:
    return any(_type_contains_typevar(param.type_expr, ctx) for param in module.params) or _type_contains_typevar(
        module.return_type_expr, ctx
    )


def _store_typed_module(state: _Tc2, module: AxonDefinition) -> None:
    existing = state.typed_modules.get(module.name)
    if (
        existing is not None
        and _definition_signature(existing) != _definition_signature(module)
    ):
        state.specialization_conflicts.add(module.name)
        return
    state.typed_modules[module.name] = module


def _normalize_typed_module(module: AxonDefinition, ctx: _TcCtx) -> AxonDefinition:
    normalized_params = tuple(
        replace(param, type_expr=_normalize_type_expr_for_module(param.type_expr, ctx))
        for param in module.params
    )
    normalized_params = tuple(
        replace(
            normalized_param,
            type_expr=_preserve_compound_dim_binders(
                original_param.type_expr, normalized_param.type_expr
            ),
        )
        for original_param, normalized_param in zip(
            module.params, normalized_params, strict=True
        )
    )
    normalized_return_type = _preserve_compound_dim_binders(
        module.return_type_expr,
        _normalize_type_expr_for_module(module.return_type_expr, ctx),
    )
    normalized = replace(
        module,
        params=normalized_params,
        return_type_expr=normalized_return_type,
        body_expr=_normalize_expr(module.body_expr, ctx) if module.body_expr is not None else None,
        statements=tuple(_normalize_statement(stmt, ctx) for stmt in module.statements),
    )
    replacements: dict[str, AxonExpr] = {}
    prefix_bindings: list[AxonStatement] = []
    bound = {param.name for param in normalized.params}
    if normalized.path_param is not None:
        bound.add(normalized.path_param)
    bound.update(normalized.path_params)
    protected_dim_names = _env_dim_names(_module_header_env(normalized, ctx))
    if normalized.return_type_expr is not None:
        protected_dim_names.update(_collect_dim_names(normalized.return_type_expr))
    for name, mapped in ctx.dim_substitutions.items():
        if isinstance(mapped, tuple) or not isinstance(name, str) or name.startswith(".."):
            continue
        if (
            name in bound
            and isinstance(mapped, str)
            and mapped not in bound
            and not mapped.startswith("..")
            and mapped not in protected_dim_names
        ):
            replacements[mapped] = _annotate_dim_expr(AxonExprName(name=name), ctx)
            continue
        if name in protected_dim_names:
            continue
        expr = _dim_token_to_expr(_normalize_dim_token(mapped, ctx))
        if expr is None:
            continue
        expr = _annotate_dim_expr(expr, ctx)
        body_uses_name = (
            _expr_uses_name(normalized.body_expr, name, set())
            if normalized.body_expr is not None
            else _statements_use_name(normalized.statements, name, set())
        )
        if not body_uses_name:
            continue
        if _is_atomic_expr_for_flat(expr):
            replacements[name] = expr
        else:
            prefix_bindings.append(AxonBind(targets=(name,), expr=expr))
    if normalized.body_expr is not None:
        return replace(normalized, body_expr=_rewrite_expr_names(normalized.body_expr, replacements, bound))
    return replace(
        normalized,
        statements=(
            *prefix_bindings,
            *_rewrite_statements_names(normalized.statements, replacements, bound),
        ),
    )


def _canonicalize_fresh_dim_token(dim: DimToken, state: _Tc2) -> DimToken:
    if (
        isinstance(dim, str)
        and not dim.startswith("..")
        and not _is_fresh_dim_name(dim, state)
        and not dim.startswith("__d")
    ):
        return dim
    dim = _normalize_dim_token(dim, state.ctx)
    if isinstance(dim, int):
        return dim
    if isinstance(dim, str):
        source = state.fresh_dim_sources.get(dim)
        if source is not None:
            return source
        if dim.startswith("..__d"):
            return "..S"
        if dim.startswith("__d"):
            return "D"
        return dim
    return DimExprBinary(
        op=dim.op,
        left=_canonicalize_fresh_dim_token(dim.left, state),
        right=_canonicalize_fresh_dim_token(dim.right, state),
    )


def _canonicalize_fresh_type(tp: TypeExpr | None, state: _Tc2) -> TypeExpr | None:
    if tp is None:
        return None
    tp = _expand_alias(_apply_subst(tp, state.ctx), state.ctx)
    if isinstance(tp, TypeVar):
        source = state.fresh_type_sources.get(tp.name)
        if source is not None:
            return TypeVar(source)
        if tp.name.startswith("__tc"):
            return TypeVar("_T")
        return tp
    if isinstance(tp, TypeOptional):
        inner = _canonicalize_fresh_type(tp.inner, state)
        assert inner is not None
        return TypeOptional(inner)
    if isinstance(tp, TypeList):
        item = _canonicalize_fresh_type(tp.item, state)
        assert item is not None
        return TypeList(item)
    if isinstance(tp, TypeTuple):
        return TypeTuple(
            items=tuple(
                item
                for item in (_canonicalize_fresh_type(item, state) for item in tp.items)
                if item is not None
            )
        )
    if isinstance(tp, TypeTensor):
        return TypeTensor(
            base=tp.base,
            dims=tuple(_canonicalize_fresh_dim_token(dim, state) for dim in tp.dims),
        )
    if isinstance(tp, TypeNamed):
        return TypeNamed(
            name=tp.name,
            args=tuple(_canonicalize_fresh_dim_token(dim, state) for dim in tp.args),
        )
    return tp


def _canonicalize_fresh_expr(
    expr: AxonExpr,
    state: _Tc2,
    *,
    bound: set[str] | None = None,
) -> AxonExpr:
    bound = set(bound or ())
    inferred = _canonicalize_fresh_type(expr.inferred_type, state)
    inferred_dims = (
        tuple(_canonicalize_fresh_dim_token(dim, state) for dim in expr.inferred_dims)
        if expr.inferred_dims is not None
        else None
    )
    if isinstance(expr, AxonExprName):
        if expr.name not in bound:
            mapped = _canonicalize_fresh_dim_token(expr.name, state)
            if isinstance(mapped, str) and mapped != expr.name and not mapped.startswith(".."):
                expr = replace(expr, name=mapped)
    if isinstance(expr, AxonExprAscribe):
        return replace(
            expr,
            expr=_canonicalize_fresh_expr(expr.expr, state, bound=bound),
            type_expr=_canonicalize_fresh_type(expr.type_expr, state) or TypeAny(),
            inferred_type=inferred,
            inferred_dims=inferred_dims,
        )
    if isinstance(expr, AxonExprBinary):
        expr = replace(
            expr,
            left=_canonicalize_fresh_expr(expr.left, state, bound=bound),
            right=_canonicalize_fresh_expr(expr.right, state, bound=bound),
        )
    elif isinstance(expr, AxonExprCall):
        expr = replace(
            expr,
            args=tuple(_canonicalize_fresh_expr(arg, state, bound=bound) for arg in expr.args),
            kwargs={
                key: _canonicalize_fresh_expr(value, state, bound=bound)
                if isinstance(value, AxonExpr)
                else value
                for key, value in expr.kwargs.items()
            },
        )
    elif isinstance(expr, AxonExprList):
        expr = replace(
            expr,
            items=tuple(_canonicalize_fresh_expr(item, state, bound=bound) for item in expr.items),
        )
    elif isinstance(expr, AxonExprTuple):
        expr = replace(
            expr,
            items=tuple(_canonicalize_fresh_expr(item, state, bound=bound) for item in expr.items),
        )
    elif isinstance(expr, AxonExprTernary | AxonExprIf):
        expr = replace(
            expr,
            cond=_canonicalize_fresh_expr(expr.cond, state, bound=bound),
            true_expr=_canonicalize_fresh_expr(expr.true_expr, state, bound=bound),
            false_expr=_canonicalize_fresh_expr(expr.false_expr, state, bound=bound),
        )
    elif isinstance(expr, AxonExprParen):
        expr = replace(expr, inner=_canonicalize_fresh_expr(expr.inner, state, bound=bound))
    elif isinstance(expr, AxonExprDo):
        expr = replace(
            expr,
            body=tuple(_canonicalize_fresh_statement(stmt, state, bound=bound) for stmt in expr.body),
        )
    elif isinstance(expr, AxonExprBind):
        expr = replace(
            expr,
            value=_canonicalize_fresh_expr(expr.value, state, bound=bound),
            body=_canonicalize_fresh_expr(expr.body, state, bound={*bound, expr.var}),
        )
    elif isinstance(expr, AxonExprPipe):
        expr = replace(
            expr,
            value=_canonicalize_fresh_expr(expr.value, state, bound=bound),
            stages=tuple(
                _canonicalize_fresh_expr(stage, state, bound=bound) for stage in expr.stages
            ),
        )
    elif isinstance(expr, AxonExprLambda):
        expr = replace(expr, body=_canonicalize_fresh_expr(expr.body, state, bound={*bound, expr.var}))
    return replace(expr, inferred_type=inferred, inferred_dims=inferred_dims)


def _canonicalize_fresh_statement(
    stmt: AxonStatement,
    state: _Tc2,
    *,
    bound: set[str] | None = None,
) -> AxonStatement:
    bound = set(bound or ())
    if isinstance(stmt, AxonBind):
        return replace(stmt, expr=_canonicalize_fresh_expr(stmt.expr, state, bound=bound))
    if isinstance(stmt, AxonReturn | AxonYield):
        return replace(
            stmt,
            values=tuple(
                _canonicalize_fresh_expr(value, state, bound=bound) for value in stmt.values
            ),
        )
    if isinstance(stmt, AxonCond):
        return replace(
            stmt,
            cond=_canonicalize_fresh_expr(stmt.cond, state, bound=bound),
            true_body=tuple(
                _canonicalize_fresh_statement(item, state, bound=bound)
                for item in stmt.true_body
            ),
            false_body=tuple(
                _canonicalize_fresh_statement(item, state, bound=bound)
                for item in stmt.false_body
            ),
        )
    return stmt


def _canonicalize_fresh_statements(
    statements: tuple[AxonStatement, ...],
    state: _Tc2,
    *,
    bound: set[str],
) -> tuple[AxonStatement, ...]:
    out: list[AxonStatement] = []
    local = set(bound)
    for stmt in statements:
        out.append(_canonicalize_fresh_statement(stmt, state, bound=local))
        if isinstance(stmt, AxonBind):
            local.update(name for name in stmt.targets if name != "_")
        elif isinstance(stmt, AxonRepeat):
            targets = stmt.targets or ()
            local.update(name for name in targets if name != "_")
        elif isinstance(stmt, AxonScopeBind):
            local.update(name for name in stmt.targets if name != "_")
    return tuple(out)


def _canonicalize_fresh_module(module: AxonDefinition, state: _Tc2) -> AxonDefinition:
    bound = {
        *(param.name for param in module.params),
        *module.path_params,
    }
    if module.path_param is not None:
        bound.add(module.path_param)
    canonical_params = tuple(
        replace(param, type_expr=_canonicalize_fresh_type(param.type_expr, state))
        for param in module.params
    )
    canonical_params = tuple(
        replace(
            canonical_param,
            type_expr=_preserve_compound_dim_binders(
                original_param.type_expr, canonical_param.type_expr
            ),
        )
        for original_param, canonical_param in zip(
            module.params, canonical_params, strict=True
        )
    )
    canonical_return_type = _preserve_compound_dim_binders(
        module.return_type_expr,
        _canonicalize_fresh_type(module.return_type_expr, state),
    )
    return replace(
        module,
        params=canonical_params,
        return_type_expr=canonical_return_type,
        body_expr=_canonicalize_fresh_expr(module.body_expr, state, bound=bound)
        if module.body_expr is not None
        else None,
        statements=_canonicalize_fresh_statements(module.statements, state, bound=bound),
    )


def _prefer_closed_constant_dim_names(state: _Tc2) -> None:
    constant_names = {
        name
        for name, module in state.modules_by_name.items()
        if not module.params and not module.path_params and module.path_param is None
    }
    for name, mapped in list(state.ctx.dim_substitutions.items()):
        if (
            name in constant_names
            and isinstance(mapped, str)
            and mapped not in constant_names
            and not mapped.startswith("..")
        ):
            state.ctx.dim_substitutions.pop(name, None)
            _bind_dim_name(mapped, name, state.ctx)


def _align_loop_continue_signatures(
    typed_by_name: dict[str, AxonDefinition],
) -> dict[str, AxonDefinition]:
    out = dict(typed_by_name)
    for name, module in typed_by_name.items():
        if "_recur_continue_" not in name:
            continue
        prefix = name.split("_recur_continue_", 1)[0]
        recur_candidates = [
            candidate
            for candidate_name, candidate in typed_by_name.items()
            if candidate_name.startswith(f"{prefix}_recur_")
            and "_recur_continue_" not in candidate_name
        ]
        if len(recur_candidates) != 1:
            continue
        recur = recur_candidates[0]
        if len(module.params) != len(recur.params):
            continue
        out[name] = replace(
            module,
            params=tuple(
                replace(param, type_expr=recur_param.type_expr)
                for param, recur_param in zip(module.params, recur.params, strict=True)
            ),
            return_type_expr=recur.return_type_expr,
        )
    return out


def _affine_add(
    left: tuple[dict[str, Fraction], Fraction],
    right: tuple[dict[str, Fraction], Fraction],
    scale: Fraction = Fraction(1),
) -> tuple[dict[str, Fraction], Fraction]:
    coeffs = dict(left[0])
    for name, coeff in right[0].items():
        updated = coeffs.get(name, Fraction(0)) + (scale * coeff)
        if updated:
            coeffs[name] = updated
        else:
            coeffs.pop(name, None)
    return coeffs, left[1] + (scale * right[1])


def _affine_scale(
    value: tuple[dict[str, Fraction], Fraction], scale: Fraction
) -> tuple[dict[str, Fraction], Fraction]:
    if scale == 0:
        return {}, Fraction(0)
    return {name: coeff * scale for name, coeff in value[0].items()}, value[1] * scale


def _dim_affine(dim: DimToken, ctx: _TcCtx) -> tuple[dict[str, Fraction], Fraction] | None:
    dim = _normalize_dim_token(dim, ctx)
    if isinstance(dim, int):
        return {}, Fraction(dim)
    if isinstance(dim, str):
        return {dim: Fraction(1)}, Fraction(0)
    left = _dim_affine(dim.left, ctx)
    right = _dim_affine(dim.right, ctx)
    if dim.op == "+" and left is not None and right is not None:
        return _affine_add(left, right)
    if dim.op == "-" and left is not None and right is not None:
        return _affine_add(left, right, Fraction(-1))
    if dim.op == "*":
        if left is not None and not left[0] and right is not None:
            return _affine_scale(right, left[1])
        if right is not None and not right[0] and left is not None:
            return _affine_scale(left, right[1])
        return None
    if dim.op == "/" and left is not None and right is not None and not right[0] and right[1] != 0:
        return _affine_scale(left, Fraction(1, 1) / right[1])
    return None


def _dim_equivalent(left: DimToken, right: DimToken, ctx: _TcCtx) -> bool:
    left = _normalize_dim_token(left, ctx)
    right = _normalize_dim_token(right, ctx)
    if left == right:
        return True
    left_affine = _dim_affine(left, ctx)
    right_affine = _dim_affine(right, ctx)
    return left_affine is not None and right_affine is not None and left_affine == right_affine


def _all_fresh_dim_names(dim: DimToken, state: _Tc2) -> bool:
    names = dim_token_names(dim)
    return bool(names) and all(_is_fresh_dim_name(name, state) for name in names)


def _unify_return_dim(
    actual: DimToken,
    expected: DimToken,
    state: _Tc2,
    protected_dim_names: set[str],
) -> DimToken:
    ctx = state.ctx
    actual = _normalize_dim_token(actual, ctx)
    expected = _normalize_dim_token(expected, ctx)
    if _dim_equivalent(actual, expected, ctx):
        return expected
    if isinstance(expected, str) and not expected.startswith("..") and _is_fresh_dim_name(expected, state):
        _bind_dim_name(expected, actual, ctx)
        return _normalize_dim_token(actual, ctx)
    if isinstance(actual, str) and not actual.startswith("..") and _is_fresh_dim_name(actual, state):
        _bind_dim_name(actual, expected, ctx)
        return _normalize_dim_token(expected, ctx)
    if _all_fresh_dim_names(expected, state):
        return actual
    if _all_fresh_dim_names(actual, state):
        return expected
    if (
        isinstance(actual, str)
        and not actual.startswith("..")
        and actual in protected_dim_names
        and not _is_fresh_dim_name(actual, state)
        and actual in dim_token_names(expected)
    ):
        return actual
    if (
        isinstance(expected, str)
        and not expected.startswith("..")
        and expected in protected_dim_names
        and not _is_fresh_dim_name(expected, state)
    ):
        if isinstance(actual, str) and not actual.startswith(".."):
            _bind_dim_name(actual, expected, ctx)
        return expected
    try:
        unified = _unify(
            TypeTensor(base="Tensor", dims=(actual,)), TypeTensor(base="Tensor", dims=(expected,)), ctx
        )
    except ValueError:
        if _dim_equivalent(actual, expected, ctx):
            return expected
        raise
    if isinstance(unified, TypeTensor) and len(unified.dims) == 1:
        return unified.dims[0]
    raise ValueError(f"Axon typecheck2 failed: return dim mismatch {actual!r} vs {expected!r}")


def _unify_return_type(
    actual: TypeExpr,
    expected: TypeExpr,
    state: _Tc2,
    protected_dim_names: set[str],
) -> TypeExpr:
    ctx = state.ctx
    actual = _expand_alias(_apply_subst(actual, ctx), ctx)
    expected = _expand_alias(_apply_subst(expected, ctx), ctx)
    if isinstance(expected, TypeAny):
        return _apply_subst(actual, ctx)
    if isinstance(actual, TypeTensor) and isinstance(expected, TypeTensor):
        if any(isinstance(dim, str) and dim.startswith("..") for dim in (*actual.dims, *expected.dims)):
            _unify_tc2(actual, expected, ctx)
            return _apply_subst(expected, ctx)
        if len(actual.dims) != len(expected.dims):
            raise ValueError("Axon typecheck2 failed: return tensor rank mismatch")
        return TypeTensor(
            base=expected.base,
            dims=tuple(
                _unify_return_dim(actual_dim, expected_dim, state, protected_dim_names)
                for actual_dim, expected_dim in zip(actual.dims, expected.dims, strict=True)
            ),
        )
    if isinstance(actual, TypeTuple) and isinstance(expected, TypeTuple):
        if len(actual.items) != len(expected.items):
            raise ValueError("Axon typecheck2 failed: return tuple arity mismatch")
        return TypeTuple(
            items=tuple(
                _unify_return_type(actual_item, expected_item, state, protected_dim_names)
                for actual_item, expected_item in zip(actual.items, expected.items, strict=True)
            )
        )
    if isinstance(actual, TypeOptional) and isinstance(expected, TypeOptional):
        return TypeOptional(_unify_return_type(actual.inner, expected.inner, state, protected_dim_names))
    if isinstance(actual, TypeNull) and isinstance(expected, TypeOptional):
        _unify_tc2(actual, expected, ctx)
        return _apply_subst(expected, ctx)
    if isinstance(actual, TypeList) and isinstance(expected, TypeList):
        return TypeList(_unify_return_type(actual.item, expected.item, state, protected_dim_names))
    _unify_tc2(actual, expected, ctx)
    return _apply_subst(expected, ctx)


def _resolve_return_dim_token(
    dim: DimToken,
    expr_defs: dict[str, AxonExpr],
    ctx: _TcCtx,
    seen: frozenset[str] = frozenset(),
) -> DimToken:
    dim = _normalize_dim_token(dim, ctx)
    if isinstance(dim, str):
        if dim in seen:
            return dim
        resolved = expr_defs.get(dim)
        if resolved is None:
            return dim
        token = _expr_to_dim_token_resolved(resolved, expr_defs, seen | {dim})
        if token is None:
            return dim
        return _resolve_return_dim_token(token, expr_defs, ctx, seen | {dim})
    if isinstance(dim, DimExprBinary):
        return _normalize_dim_token(
            DimExprBinary(
                op=dim.op,
                left=_resolve_return_dim_token(dim.left, expr_defs, ctx, seen),
                right=_resolve_return_dim_token(dim.right, expr_defs, ctx, seen),
            ),
            ctx,
        )
    return dim


def _resolve_return_type_dim_aliases(
    tp: TypeExpr,
    expr_defs: dict[str, AxonExpr],
    ctx: _TcCtx,
) -> TypeExpr:
    tp = _apply_subst(tp, ctx)
    if isinstance(tp, TypeTensor):
        return TypeTensor(
            base=tp.base,
            dims=tuple(_resolve_return_dim_token(dim, expr_defs, ctx) for dim in tp.dims),
        )
    if isinstance(tp, TypeTuple):
        return TypeTuple(
            items=tuple(_resolve_return_type_dim_aliases(item, expr_defs, ctx) for item in tp.items)
        )
    if isinstance(tp, TypeOptional):
        return TypeOptional(_resolve_return_type_dim_aliases(tp.inner, expr_defs, ctx))
    if isinstance(tp, TypeList):
        return TypeList(_resolve_return_type_dim_aliases(tp.item, expr_defs, ctx))
    return tp


def _bind_call_args(
    state: _Tc2,
    module: AxonDefinition,
    args: tuple[AxonExpr, ...],
    kwargs: dict[str, AxonKwargValue],
    caller_env: dict[str, TypeExpr],
    expr_defs: dict[str, AxonExpr],
) -> tuple[_CallBinding, list[AxonExpr], dict[str, AxonKwargValue]]:
    typed_args: list[AxonExpr] = []
    typed_kwargs: dict[str, AxonKwargValue] = {}
    param_types: dict[str, TypeExpr] = {}
    path_types: dict[str, TypeExpr] = {}
    bound_expr_defs: dict[str, AxonExpr] = {}
    fresh_dim_sources_before = set(state.fresh_dim_sources)
    instantiated_param_types, return_types = _instantiate_call_signature(module, state)

    def bind_fresh_dims_from_actual(param_name: str, actual: AxonExpr) -> None:
        resolved_actual = _resolved_expr_def(actual, expr_defs)
        token = _expr_to_dim_token_resolved(resolved_actual, expr_defs)
        if token is None:
            return
        for fresh_name, source_name in tuple(state.fresh_dim_sources.items()):
            if source_name == param_name:
                _bind_dim_name(fresh_name, token, state.ctx)

    positional = list(args)
    path_params = list(module.path_params)
    if module.path_param and module.path_param not in path_params:
        path_params.append(module.path_param)
    for name in path_params:
        if not positional:
            raise ValueError(f"Axon typecheck2 failed: missing path argument for {module.name}.{name}")
        typed, tp = _tc_expr(state, positional.pop(0), caller_env, expr_defs)
        typed_args.append(typed)
        _unify(tp, TypePath(), state.ctx)
        path_types[name] = TypePath()

    for param, formal_env_type in zip(module.params, instantiated_param_types, strict=True):
        actual_expr: AxonExpr | None = None
        actual_is_positional = False
        if positional:
            actual_expr = positional.pop(0)
            actual_is_positional = True
        else:
            raw_kwarg = kwargs.get(param.name)
            if isinstance(raw_kwarg, AxonExpr):
                actual_expr = raw_kwarg
            elif raw_kwarg is not None:
                actual_expr = _scalar_to_expr(raw_kwarg)
        if actual_expr is None:
            if param.default_expr is not None:
                default_env = {**caller_env, **param_types, **path_types}
                typed_default, actual_tp = _tc_expr(state, param.default_expr, default_env, expr_defs)
                try:
                    bound_tp = _unify_tc2(actual_tp, formal_env_type, state.ctx)
                except Exception as exc:
                    raise ValueError(
                        f"Axon typecheck2 failed: argument {param.name!r} for {module.name!r} "
                        f"has type {actual_tp!r} but expected {formal_env_type!r}"
                    ) from exc
                bound_expr_defs[param.name] = _resolved_expr_def_deep(typed_default, expr_defs)
                bind_fresh_dims_from_actual(param.name, typed_default)
            elif param.optional:
                actual_tp = TypeNull()
                try:
                    bound_tp = _unify_tc2(actual_tp, formal_env_type, state.ctx)
                except Exception as exc:
                    raise ValueError(
                        f"Axon typecheck2 failed: argument {param.name!r} for {module.name!r} "
                        f"has type {actual_tp!r} but expected {formal_env_type!r}"
                    ) from exc
            else:
                raise ValueError(f"Axon typecheck2 failed: missing argument {param.name!r} for {module.name}")
        else:
            typed_actual, actual_tp = _tc_expr(state, actual_expr, caller_env, expr_defs)
            try:
                bound_tp = _unify_tc2(actual_tp, formal_env_type, state.ctx)
            except Exception as exc:
                raise ValueError(
                    f"Axon typecheck2 failed: argument {param.name!r} for {module.name!r} "
                    f"has type {actual_tp!r} but expected {formal_env_type!r}"
                ) from exc
            if actual_is_positional:
                typed_args.append(typed_actual)
            else:
                typed_kwargs[param.name] = typed_actual
            bound_expr_defs[param.name] = _resolved_expr_def_deep(typed_actual, expr_defs)
            bind_fresh_dims_from_actual(param.name, typed_actual)
        param_types[param.name] = _apply_subst(bound_tp, state.ctx)

    if positional:
        raise ValueError(f"Axon typecheck2 failed: too many arguments for {module.name}")
    for key, value in kwargs.items():
        if key in {param.name for param in module.params}:
            continue
        typed_kwargs[key] = value
    dim_bindings: dict[str, DimToken] = {}
    for fresh_name, source_name in state.fresh_dim_sources.items():
        if fresh_name in fresh_dim_sources_before or source_name.startswith(".."):
            continue
        value = _normalize_dim_token(fresh_name, state.ctx)
        if value == fresh_name or value == source_name:
            continue
        if source_name in dim_token_names(value):
            continue
        dim_bindings[source_name] = value
    return (
        _CallBinding(
            tuple(typed_args),
            typed_kwargs,
            param_types,
            path_types,
            return_types,
            bound_expr_defs,
            dim_bindings,
        ),
        typed_args,
        typed_kwargs,
    )


def _scalar_to_expr(value: Any) -> AxonExpr:
    if isinstance(value, bool):
        return AxonExprBool(value=value)
    if isinstance(value, int):
        return AxonExprInt(value=value)
    if isinstance(value, float):
        return AxonExprFloat(value=value)
    if isinstance(value, str):
        return AxonExprString(value=value)
    if value is None:
        return AxonExprNull()
    raise ValueError(f"Axon typecheck2 failed: unsupported scalar kwarg {value!r}")


def _resolved_expr_def(expr: AxonExpr, expr_defs: dict[str, AxonExpr]) -> AxonExpr:
    if isinstance(expr, AxonExprAscribe | AxonExprParen):
        inner = expr.expr if isinstance(expr, AxonExprAscribe) else expr.inner
        resolved = _resolved_expr_def(inner, expr_defs)
        return (
            replace(expr, expr=resolved)
            if isinstance(expr, AxonExprAscribe)
            else replace(expr, inner=resolved)
        )
    if isinstance(expr, AxonExprName):
        return expr_defs.get(expr.name, expr)
    return expr


def _resolved_expr_def_deep(
    expr: AxonExpr,
    expr_defs: dict[str, AxonExpr],
    seen: frozenset[str] = frozenset(),
) -> AxonExpr:
    if isinstance(expr, AxonExprAscribe):
        return replace(expr, expr=_resolved_expr_def_deep(expr.expr, expr_defs, seen))
    if isinstance(expr, AxonExprParen):
        return replace(expr, inner=_resolved_expr_def_deep(expr.inner, expr_defs, seen))
    if isinstance(expr, AxonExprName):
        if expr.name in seen:
            return expr
        resolved = expr_defs.get(expr.name)
        if resolved is None:
            return expr
        return _resolved_expr_def_deep(resolved, expr_defs, seen | {expr.name})
    if isinstance(expr, AxonExprBinary):
        return replace(
            expr,
            left=_resolved_expr_def_deep(expr.left, expr_defs, seen),
            right=_resolved_expr_def_deep(expr.right, expr_defs, seen),
        )
    if isinstance(expr, AxonExprCall):
        return replace(
            expr,
            args=tuple(_resolved_expr_def_deep(arg, expr_defs, seen) for arg in expr.args),
            kwargs={
                key: _resolved_expr_def_deep(value, expr_defs, seen) if isinstance(value, AxonExpr) else value
                for key, value in expr.kwargs.items()
            },
        )
    if isinstance(expr, AxonExprTuple):
        return replace(
            expr,
            items=tuple(_resolved_expr_def_deep(item, expr_defs, seen) for item in expr.items),
        )
    if isinstance(expr, AxonExprList):
        return replace(
            expr,
            items=tuple(_resolved_expr_def_deep(item, expr_defs, seen) for item in expr.items),
        )
    if isinstance(expr, AxonExprTernary | AxonExprIf):
        return replace(
            expr,
            cond=_resolved_expr_def_deep(expr.cond, expr_defs, seen),
            true_expr=_resolved_expr_def_deep(expr.true_expr, expr_defs, seen),
            false_expr=_resolved_expr_def_deep(expr.false_expr, expr_defs, seen),
        )
    return expr


def _resolved_atomic_expr_def(expr: AxonExpr, expr_defs: dict[str, AxonExpr]) -> AxonExpr:
    current = expr
    seen: set[str] = set()
    wrappers: list[type[AxonExprAscribe] | type[AxonExprParen]] = []
    while isinstance(current, AxonExprAscribe | AxonExprParen):
        wrappers.append(type(current))
        current = current.expr if isinstance(current, AxonExprAscribe) else current.inner
    if isinstance(current, AxonExprName):
        while isinstance(current, AxonExprName) and current.name not in seen:
            seen.add(current.name)
            resolved = expr_defs.get(current.name)
            if resolved is None:
                break
            stripped = resolved
            while isinstance(stripped, AxonExprAscribe | AxonExprParen):
                stripped = stripped.expr if isinstance(stripped, AxonExprAscribe) else stripped.inner
            if _is_atomic_expr_for_flat(stripped):
                current = stripped
                continue
            break
    out: AxonExpr = current
    for wrapper in reversed(wrappers):
        out = AxonExprParen(inner=out) if wrapper is AxonExprParen else out
    return out


def _expr_to_dim_token_for_type_rule(
    state: _Tc2,
    expr: AxonExpr,
    expr_defs: dict[str, AxonExpr],
    seen: frozenset[str] = frozenset(),
) -> DimToken | None:
    resolved = _resolved_atomic_expr_def(expr, expr_defs)
    while isinstance(resolved, AxonExprAscribe | AxonExprParen):
        resolved = resolved.expr if isinstance(resolved, AxonExprAscribe) else resolved.inner
    if isinstance(resolved, AxonExprNull):
        return None
    if isinstance(resolved, AxonExprInt):
        return resolved.value
    if isinstance(resolved, AxonExprName):
        if resolved.name in seen:
            return resolved.name
        target = expr_defs.get(resolved.name)
        if target is None:
            return _normalize_dim_token(resolved.name, state.ctx)
        return _expr_to_dim_token_for_type_rule(state, target, expr_defs, seen | {resolved.name})
    if isinstance(resolved, AxonExprCall):
        module = state.modules_by_name.get(resolved.callee)
        non_path_params = (
            [
                param
                for param in module.params
                if not isinstance(
                    _expand_alias(_apply_subst(param.type_expr or TypeAny(), state.ctx), state.ctx),
                    TypePath,
                )
            ]
            if module is not None
            else None
        )
        stripped_args = [
            arg.expr
            if isinstance(arg, AxonExprAscribe)
            else arg.inner
            if isinstance(arg, AxonExprParen)
            else arg
            for arg in resolved.args
        ]
        if (
            module is not None
            and not non_path_params
            and all(isinstance(arg, AxonExprPath) for arg in stripped_args)
            and not resolved.kwargs
        ):
            return resolved.callee
        expr_type = _expand_alias(_apply_subst(resolved.inferred_type or TypeAny(), state.ctx), state.ctx)
        if isinstance(expr_type, TypeInt | TypeDim) and not resolved.args and not resolved.kwargs:
            return resolved.callee
        return None
    if isinstance(resolved, AxonExprBinary) and resolved.op in {"+", "-", "*", "/"}:
        left = _expr_to_dim_token_for_type_rule(state, resolved.left, expr_defs, seen)
        right = _expr_to_dim_token_for_type_rule(state, resolved.right, expr_defs, seen)
        if left is None or right is None:
            return None
        return _normalize_dim_token(DimExprBinary(op=resolved.op, left=left, right=right), state.ctx)
    return _expr_to_dim_token_resolved(resolved, expr_defs)


def _expr_contains_null_for_required_type(
    expr: AxonExpr,
    expected: TypeExpr,
    expr_defs: dict[str, AxonExpr],
    ctx: _TcCtx,
) -> bool:
    expected = _expand_alias(_apply_subst(expected, ctx), ctx)
    if isinstance(expected, TypeAny | TypeOptional):
        return False
    resolved = _resolved_atomic_expr_def(expr, expr_defs)
    while isinstance(resolved, AxonExprAscribe | AxonExprParen):
        resolved = resolved.expr if isinstance(resolved, AxonExprAscribe) else resolved.inner
    if isinstance(resolved, AxonExprNull):
        return True
    if isinstance(expected, TypeList) and isinstance(resolved, AxonExprList):
        return any(
            _expr_contains_null_for_required_type(item, expected.item, expr_defs, ctx)
            for item in resolved.items
        )
    if isinstance(expected, TypeTuple) and isinstance(resolved, AxonExprTuple):
        return any(
            _expr_contains_null_for_required_type(item, item_expected, expr_defs, ctx)
            for item, item_expected in zip(resolved.items, expected.items, strict=False)
        )
    return False


def _static_bool(expr: AxonExpr, expr_defs: dict[str, AxonExpr]) -> bool | None:
    expr = _resolved_expr_def(expr, expr_defs)
    while isinstance(expr, AxonExprAscribe | AxonExprParen):
        expr = expr.expr if isinstance(expr, AxonExprAscribe) else expr.inner
        expr = _resolved_expr_def(expr, expr_defs)
    if isinstance(expr, AxonExprBool):
        return expr.value
    if not isinstance(expr, AxonExprBinary) or expr.op not in {"==", "!="}:
        return None
    left = _resolved_expr_def(expr.left, expr_defs)
    right = _resolved_expr_def(expr.right, expr_defs)
    while isinstance(left, AxonExprAscribe | AxonExprParen):
        left = left.expr if isinstance(left, AxonExprAscribe) else left.inner
        left = _resolved_expr_def(left, expr_defs)
    while isinstance(right, AxonExprAscribe | AxonExprParen):
        right = right.expr if isinstance(right, AxonExprAscribe) else right.inner
        right = _resolved_expr_def(right, expr_defs)
    if isinstance(left, AxonExprNull) or isinstance(right, AxonExprNull):
        if not isinstance(left, AxonExprNull) and not isinstance(
            left,
            AxonExprInt | AxonExprFloat | AxonExprBool | AxonExprString | AxonExprPath,
        ):
            return None
        if not isinstance(right, AxonExprNull) and not isinstance(
            right,
            AxonExprInt | AxonExprFloat | AxonExprBool | AxonExprString | AxonExprPath,
        ):
            return None
        equal = isinstance(left, AxonExprNull) and isinstance(right, AxonExprNull)
        return equal if expr.op == "==" else not equal
    if isinstance(left, AxonExprBool) and isinstance(right, AxonExprBool):
        equal = left.value == right.value
        return equal if expr.op == "==" else not equal
    return None


def _null_checked_name(expr: AxonExpr) -> tuple[str, bool] | None:
    while isinstance(expr, AxonExprAscribe | AxonExprParen):
        expr = expr.expr if isinstance(expr, AxonExprAscribe) else expr.inner
    if not isinstance(expr, AxonExprBinary) or expr.op not in {"==", "!="}:
        return None
    left = expr.left
    right = expr.right
    while isinstance(left, AxonExprAscribe | AxonExprParen):
        left = left.expr if isinstance(left, AxonExprAscribe) else left.inner
    while isinstance(right, AxonExprAscribe | AxonExprParen):
        right = right.expr if isinstance(right, AxonExprAscribe) else right.inner
    if isinstance(left, AxonExprName) and isinstance(right, AxonExprNull):
        return left.name, expr.op == "=="
    if isinstance(right, AxonExprName) and isinstance(left, AxonExprNull):
        return right.name, expr.op == "=="
    return None


def _branch_envs(
    cond_expr: AxonExpr, env: dict[str, TypeExpr], expr_defs: dict[str, AxonExpr]
) -> tuple[dict[str, TypeExpr], dict[str, TypeExpr]]:
    true_env = dict(env)
    false_env = dict(env)
    checked = _null_checked_name(cond_expr)
    if checked is None:
        checked = _null_checked_name(_resolved_expr_def(cond_expr, expr_defs))
    if checked is None:
        return true_env, false_env
    name, true_when_null = checked
    current = env.get(name)
    if not isinstance(current, TypeOptional):
        return true_env, false_env
    if true_when_null:
        true_env[name] = TypeNull()
        false_env[name] = current.inner
    else:
        true_env[name] = current.inner
        false_env[name] = TypeNull()
    return true_env, false_env


def _env_dim_names(env: dict[str, TypeExpr]) -> set[str]:
    names: set[str] = set()
    for tp in env.values():
        names.update(_collect_dim_names(tp))
    return names


def _tc_primitive_call(
    state: _Tc2,
    expr: AxonExprCall,
    typed_args: list[AxonExpr],
    arg_types: list[TypeExpr],
    typed_kwargs: dict[str, AxonKwargValue],
    kwarg_types: dict[str, TypeExpr],
    expr_defs: dict[str, AxonExpr],
) -> tuple[AxonExpr, TypeExpr] | None:
    op_name = _primitive_op_name(expr.callee)
    if op_name is None:
        return None
    signature = get_op_lowering_type_signature(op_name)
    if signature is None:
        return None
    arg_specs = signature.get("args")
    if isinstance(arg_specs, tuple):
        for idx, (arg_type, spec) in enumerate(zip(arg_types, arg_specs, strict=False)):
            if isinstance(spec, str):
                expected = _type_expr_from_spec(spec, ctx=state.ctx, module_name=f"_op::{op_name}")
                if idx < len(typed_args) and _expr_contains_null_for_required_type(
                    typed_args[idx], expected, expr_defs, state.ctx
                ):
                    raise ValueError(
                        f"Axon typecheck2 failed: primitive {expr.callee} argument {idx} "
                        f"requires {spec}, got null"
                    )
                _unify_tc2(arg_type, expected, state.ctx)
    kwarg_specs = signature.get("kwargs")
    if isinstance(kwarg_specs, dict):
        for key, kwarg_type in kwarg_types.items():
            spec = kwarg_specs.get(key)
            if isinstance(spec, str):
                expected = _type_expr_from_spec(spec, ctx=state.ctx, module_name=f"_op::{op_name}")
                raw_kwarg = typed_kwargs.get(key)
                if isinstance(raw_kwarg, AxonExpr) and _expr_contains_null_for_required_type(
                    raw_kwarg, expected, expr_defs, state.ctx
                ):
                    raise ValueError(
                        f"Axon typecheck2 failed: primitive {expr.callee} kwarg {key!r} "
                        f"requires {spec}, got null"
                    )
                _unify_tc2(kwarg_type, expected, state.ctx)
    typed_call = AxonExprCall(callee=expr.callee, args=tuple(typed_args), kwargs=typed_kwargs)
    type_rule_args = tuple(_resolved_expr_def_deep(arg, expr_defs) for arg in typed_args)
    type_rule_kwargs = {
        key: _resolved_expr_def_deep(value, expr_defs) if isinstance(value, AxonExpr) else value
        for key, value in typed_kwargs.items()
    }
    type_rule = get_op_type_rule(op_name)
    if type_rule is not None:
        inferred = type_rule(
            arg_types=tuple(arg_types),
            kwarg_types=dict(kwarg_types),
            args=type_rule_args,
            kwargs=dict(type_rule_kwargs),
            helpers=_PrimitiveTypeHelpers(
                type_dims=lambda tp: _type_dims(tp, state.ctx),
                expr_to_dim_token=lambda value: _expr_to_dim_token_for_type_rule(state, value, expr_defs)
                if isinstance(value, AxonExpr)
                else None,
                type_tensor=lambda *, dims: TypeTensor(base="Tensor", dims=tuple(dims)),
                resolve_name_expr=lambda name: expr_defs.get(name),
                broadcast_tensor_dims=lambda left, right: _unify_broadcast_tensor_dims(
                    tuple(left), tuple(right), state.ctx
                ),
            ),
        )
        if _is_type_expr_instance(inferred):
            inferred = _resolve_type_dim_aliases(inferred, expr_defs)
            arity = len(inferred.items) if isinstance(inferred, TypeTuple) else 1
            return _annotate(state.ctx, typed_call, inferred, arity=arity), inferred
    returns = signature.get("returns")
    if isinstance(returns, str) and returns != "dynamic":
        if returns == "Tensor[..R]" and len(arg_types) >= 2:
            left_dims = _type_dims(arg_types[0], state.ctx)
            right_dims = _type_dims(arg_types[1], state.ctx)
            dims = broadcast_shape(left_dims, right_dims)
            if dims is not None:
                tensor_result = TypeTensor(base="Tensor", dims=tuple(dims))
                return _annotate(state.ctx, typed_call, tensor_result), tensor_result
        result_tp = _type_expr_from_spec(returns, ctx=state.ctx, module_name=f"_op::{op_name}")
        arity = len(result_tp.items) if isinstance(result_tp, TypeTuple) else 1
        return _annotate(state.ctx, typed_call, result_tp, arity=arity), result_tp
    if isinstance(returns, tuple) and len(returns) == 1 and isinstance(returns[0], str):
        result_tp = _type_expr_from_spec(returns[0], ctx=state.ctx, module_name=f"_op::{op_name}")
        return _annotate(state.ctx, typed_call, result_tp), result_tp
    if isinstance(returns, tuple) and all(isinstance(item, str) for item in returns):
        result_tp = TypeTuple(
            items=tuple(
                _type_expr_from_spec(item, ctx=state.ctx, module_name=f"_op::{op_name}")
                for item in returns
            )
        )
        return _annotate(state.ctx, typed_call, result_tp, arity=len(result_tp.items)), result_tp
    if returns == "dynamic" and arg_types:
        result_tp = _apply_subst(arg_types[0], state.ctx)
        arity = len(result_tp.items) if isinstance(result_tp, TypeTuple) else 1
        return _annotate(state.ctx, typed_call, result_tp, arity=arity), result_tp
    return None


def _tc_user_call(
    state: _Tc2,
    expr: AxonExprCall,
    env: dict[str, TypeExpr],
    expr_defs: dict[str, AxonExpr],
    typed_args: list[AxonExpr],
    typed_kwargs: dict[str, AxonKwargValue],
) -> tuple[AxonExpr, TypeExpr] | None:
    module = state.modules_by_name.get(expr.callee)
    if module is None:
        return None
    binding, bound_args, bound_kwargs = _bind_call_args(state, module, expr.args, expr.kwargs, env, expr_defs)
    call_env = {**binding.path_types, **binding.param_types}
    if module.name in state.active:
        returns = binding.return_types if binding.return_types is not None else _return_types(module, state.ctx)
        result: TypeExpr
        if returns is None:
            result = TypeAny()
        elif len(returns) == 1:
            result = _apply_subst(returns[0], state.ctx)
        else:
            result = TypeTuple(items=tuple(_apply_subst(item, state.ctx) for item in returns))
    else:
        caller_substitutions = dict(state.ctx.substitutions)
        caller_dim_substitutions = dict(state.ctx.dim_substitutions)
        try:
            for name, value in binding.dim_bindings.items():
                _bind_dim_name(name, value, state.ctx)
            try:
                typed_module, return_tp = _tc_definition(
                    state,
                    module,
                    call_env,
                    call_expr_defs=binding.expr_defs,
                    expected_return_types=binding.return_types,
                )
            except ValueError as exc:
                raise ValueError(
                    f"Axon typecheck2 failed while checking call to {module.name!r}"
                ) from exc
            result = (
                _resolve_return_type_dim_aliases(return_tp, binding.expr_defs, state.ctx)
                if return_tp is not None
                else TypeAny()
            )
            _store_typed_module(state, typed_module)
        finally:
            state.ctx.substitutions = caller_substitutions
            state.ctx.dim_substitutions = caller_dim_substitutions
    typed_call = AxonExprCall(callee=expr.callee, args=tuple(bound_args), kwargs=bound_kwargs)
    arity = len(result.items) if isinstance(result, TypeTuple) else 1
    return _annotate(state.ctx, typed_call, result, arity=arity), result


def _tc_expr(
    state: _Tc2,
    expr: AxonExpr,
    env: dict[str, TypeExpr],
    expr_defs: dict[str, AxonExpr],
) -> tuple[AxonExpr, TypeExpr]:
    if isinstance(expr, AxonExprAscribe):
        typed_inner, inner_tp = _tc_expr(state, expr.expr, env, expr_defs)
        tp = _scoped_typevars(
            expr.type_expr,
            module_name="ascribe",
            ctx=state.ctx,
            freshen_generics=False,
            preserve_dim_names=_env_dim_names(env),
        )
        try:
            tp = _unify_return_type(inner_tp, tp, state, _env_dim_names(env))
        except ValueError as exc:
            raise ValueError(
                "Axon typecheck2 failed: type ascription mismatch "
                f"inner={inner_tp!r} ascribed={tp!r}"
            ) from exc
        return _annotate(state.ctx, replace(expr, expr=typed_inner), tp), tp
    if isinstance(expr, AxonExprParen):
        typed, tp = _tc_expr(state, expr.inner, env, expr_defs)
        return _annotate(state.ctx, replace(expr, inner=typed), tp), tp
    if isinstance(expr, AxonExprName):
        if expr.name not in env:
            raise ValueError(f"Axon typecheck2 failed: unknown name {expr.name!r}")
        tp = env[expr.name]
        return _annotate(state.ctx, expr, tp), tp
    if isinstance(expr, AxonExprInt):
        return _annotate(state.ctx, expr, TypeInt()), TypeInt()
    if isinstance(expr, AxonExprFloat):
        return _annotate(state.ctx, expr, TypeFloat()), TypeFloat()
    if isinstance(expr, AxonExprBool):
        return _annotate(state.ctx, expr, TypeBool()), TypeBool()
    if isinstance(expr, AxonExprNull):
        return _annotate(state.ctx, expr, TypeNull()), TypeNull()
    if isinstance(expr, AxonExprString):
        return _annotate(state.ctx, expr, TypeString()), TypeString()
    if isinstance(expr, AxonExprPath):
        return _annotate(state.ctx, expr, TypePath()), TypePath()
    if isinstance(expr, AxonExprList):
        typed_items: list[AxonExpr] = []
        item_tp: TypeExpr | None = None
        for item in expr.items:
            typed, tp = _tc_expr(state, item, env, expr_defs)
            typed_items.append(typed)
            item_tp = tp if item_tp is None else _join_branch_types(item_tp, tp, state.ctx)
        list_result: TypeExpr = TypeList(item=item_tp or TypeAny())
        return _annotate(state.ctx, replace(expr, items=tuple(typed_items)), list_result), list_result
    if isinstance(expr, AxonExprTuple):
        typed_items = []
        types = []
        for item in expr.items:
            typed, tp = _tc_expr(state, item, env, expr_defs)
            typed_items.append(typed)
            types.append(tp)
        tuple_result: TypeExpr = TypeTuple(items=tuple(types))
        return _annotate(state.ctx, replace(expr, items=tuple(typed_items)), tuple_result, arity=len(types)), tuple_result
    if isinstance(expr, AxonExprBinary):
        left, left_tp = _tc_expr(state, expr.left, env, expr_defs)
        right, right_tp = _tc_expr(state, expr.right, env, expr_defs)
        if expr.op in {"==", "!=", "<", "<=", ">", ">="}:
            binary_result: TypeExpr = TypeBool()
        elif expr.op in {"+", "-", "*", "/"}:
            binary_result = _binary_arithmetic_type(left_tp, right_tp, state.ctx)
        else:
            binary_result = TypeAny()
        return _annotate(state.ctx, replace(expr, left=left, right=right), binary_result), binary_result
    if isinstance(expr, AxonExprTernary | AxonExprIf):
        cond, _ = _tc_expr(state, expr.cond, env, expr_defs)
        static_cond = _static_bool(expr.cond, expr_defs)
        true_env, false_env = _branch_envs(expr.cond, env, expr_defs)
        if static_cond is True:
            true_expr, true_tp = _tc_expr(state, expr.true_expr, true_env, dict(expr_defs))
            return true_expr, true_tp
        elif static_cond is False:
            false_expr, false_tp = _tc_expr(state, expr.false_expr, false_env, dict(expr_defs))
            return false_expr, false_tp
        else:
            true_expr, true_tp = _tc_expr(state, expr.true_expr, true_env, dict(expr_defs))
            false_expr, false_tp = _tc_expr(state, expr.false_expr, false_env, dict(expr_defs))
            branch_result = _join_expr_branch_types(true_tp, false_tp, state.ctx)
        return _annotate(
            state.ctx,
            replace(expr, cond=cond, true_expr=true_expr, false_expr=false_expr),
            branch_result,
        ), branch_result
    if isinstance(expr, AxonExprCall):
        typed_args = []
        arg_types = []
        for arg in expr.args:
            typed, tp = _tc_expr(state, arg, env, expr_defs)
            typed_args.append(typed)
            arg_types.append(tp)
        typed_kwargs: dict[str, AxonKwargValue] = {}
        kwarg_types: dict[str, TypeExpr] = {}
        for key, value in expr.kwargs.items():
            if isinstance(value, AxonExpr):
                typed, tp = _tc_expr(state, value, env, expr_defs)
                typed_kwargs[key] = typed
                kwarg_types[key] = tp
            else:
                typed_kwargs[key] = value
        primitive = _tc_primitive_call(
            state, expr, typed_args, arg_types, typed_kwargs, kwarg_types, expr_defs
        )
        if primitive is not None:
            return primitive
        user = _tc_user_call(state, expr, env, expr_defs, typed_args, typed_kwargs)
        if user is not None:
            return user
        raise ValueError(f"Axon typecheck2 failed: unknown callee {expr.callee!r}")
    if isinstance(expr, AxonExprDo):
        typed_body, _, returns = _tc_statements(
            state, expr.body, dict(env), dict(expr_defs), None, _env_dim_names(env)
        )
        do_result: TypeExpr = _final_return_type(returns, state.ctx) or TypeAny()
        return _annotate(state.ctx, replace(expr, body=typed_body), do_result), do_result
    if isinstance(expr, AxonExprBind):
        typed_value, value_tp = _tc_expr(state, expr.value, env, expr_defs)
        nested_env = dict(env)
        nested_env[expr.var] = value_tp
        typed_bind_body, body_tp = _tc_expr(state, expr.body, nested_env, expr_defs)
        return _annotate(state.ctx, replace(expr, value=typed_value, body=typed_bind_body), body_tp), body_tp
    if isinstance(expr, AxonExprPipe | AxonExprLambda):
        raise ValueError(f"Axon typecheck2 failed: unsupported expression {type(expr).__name__}")
    raise TypeError(f"unsupported Axon expression: {type(expr).__name__}")


def _tc_statements(
    state: _Tc2,
    statements: tuple[AxonStatement, ...],
    env: dict[str, TypeExpr],
    expr_defs: dict[str, AxonExpr],
    expected_returns: tuple[TypeExpr, ...] | None,
    protected_dim_names: set[str],
    *,
    refine_returns_from_body: bool = False,
) -> tuple[tuple[AxonStatement, ...], dict[str, TypeExpr], tuple[TypeExpr, ...] | None]:
    typed: list[AxonStatement] = []
    returns: tuple[TypeExpr, ...] | None = None
    for stmt in statements:
        if isinstance(stmt, AxonBind):
            typed_expr, expr_tp = _tc_expr(state, stmt.expr, env, expr_defs)
            try:
                target_types = _target_type(expr_tp, len(stmt.targets), state.ctx)
            except ValueError as exc:
                raise ValueError(
                    "Axon typecheck2 failed: cannot bind expression to targets "
                    f"targets={stmt.targets!r} type={expr_tp!r}"
                ) from exc
            for name, tp in zip(stmt.targets, target_types, strict=True):
                if name == "_":
                    continue
                env[name] = _apply_subst(tp, state.ctx)
                if len(stmt.targets) == 1:
                    expr_defs[name] = typed_expr
                else:
                    expr_defs.pop(name, None)
            typed.append(replace(stmt, expr=typed_expr))
            continue
        if isinstance(stmt, AxonReturn | AxonYield):
            values = []
            value_types = []
            for value in stmt.values:
                typed_value, value_tp = _tc_expr(state, value, env, expr_defs)
                values.append(typed_value)
                value_types.append(value_tp)
            if expected_returns is not None:
                if len(value_types) == 1 and len(expected_returns) > 1:
                    value_types = list(_target_type(value_types[0], len(expected_returns), state.ctx))
                if len(value_types) != len(expected_returns):
                    raise ValueError("Axon typecheck2 failed: return arity mismatch")
                unified_values: list[TypeExpr] = []
                for actual, expected in zip(value_types, expected_returns, strict=True):
                    try:
                        unified_values.append(
                            _unify_return_type(actual, expected, state, protected_dim_names)
                        )
                    except ValueError as exc:
                        if isinstance(expected, TypeTensor) and isinstance(actual, TypeTensor):
                            unified_values.append(_apply_subst(expected, state.ctx))
                            continue
                        raise ValueError(
                            "Axon typecheck2 failed: return type mismatch "
                            f"actual={actual!r} expected={expected!r}"
                        ) from exc
                value_types = unified_values
            returns = _merge_return_types(returns, tuple(value_types), state.ctx)
            typed.append(replace(stmt, values=tuple(values)))
            continue
        if isinstance(stmt, AxonCond):
            typed_cond, _ = _tc_expr(state, stmt.cond, env, expr_defs)
            true_env, false_env = _branch_envs(stmt.cond, env, expr_defs)
            true_body, _, true_returns = _tc_statements(
                state,
                stmt.true_body,
                true_env,
                dict(expr_defs),
                expected_returns,
                protected_dim_names,
                refine_returns_from_body=refine_returns_from_body,
            )
            false_body, _, false_returns = _tc_statements(
                state,
                stmt.false_body,
                false_env,
                dict(expr_defs),
                expected_returns,
                protected_dim_names,
                refine_returns_from_body=refine_returns_from_body,
            )
            if true_returns is not None:
                returns = _merge_return_types(returns, true_returns, state.ctx)
            if false_returns is not None:
                returns = _merge_return_types(returns, false_returns, state.ctx)
            typed.append(replace(stmt, cond=typed_cond, true_body=true_body, false_body=false_body))
            continue
        if isinstance(stmt, AxonScopeBind | AxonRepeat):
            raise ValueError(f"Axon typecheck2 failed: expected flat input, got {type(stmt).__name__}")
        typed.append(stmt)
    return tuple(typed), env, returns


def _tc_definition(
    state: _Tc2,
    module: AxonDefinition,
    call_env: dict[str, TypeExpr] | None = None,
    call_expr_defs: dict[str, AxonExpr] | None = None,
    expected_return_types: tuple[TypeExpr, ...] | None = None,
) -> tuple[AxonDefinition, TypeExpr | None]:
    if module.name in state.typed_modules and call_env is None:
        typed = state.typed_modules[module.name]
        return typed, typed.return_type_expr
    specialize_signature = call_env is not None
    base_env = _module_header_env(module, state.ctx)
    if call_env is not None:
        for name, tp in call_env.items():
            base_env[name] = _apply_subst(tp, state.ctx)
    expected_returns = (
        expected_return_types
        if expected_return_types is not None
        else None
        if specialize_signature
        else _return_types(module, state.ctx)
    )
    protected_dim_names = _env_dim_names(base_env)
    signature_expr_defs = dict(call_expr_defs or {})
    signature_env = {
        name: _resolve_return_type_dim_aliases(_apply_subst(tp, state.ctx), signature_expr_defs, state.ctx)
        for name, tp in base_env.items()
    }
    signature_expected_returns = (
        tuple(
            _resolve_return_type_dim_aliases(_apply_subst(tp, state.ctx), signature_expr_defs, state.ctx)
            for tp in expected_returns
        )
        if expected_returns is not None
        else None
    )
    previous_active = state.active
    state.active = (*state.active, module.name)
    try:
        if module.body_expr is not None:
            typed_expr, expr_return_tp = _tc_expr(
                state, module.body_expr, base_env, dict(call_expr_defs or {})
            )
            if expected_returns is not None:
                if len(expected_returns) != 1:
                    raise ValueError("Axon typecheck2 failed: value definition return arity mismatch")
                expr_return_tp = _unify_return_type(
                    expr_return_tp, expected_returns[0], state, protected_dim_names
                )
            typed_module = replace(
                module,
                body_expr=typed_expr,
                return_type_expr=(
                    module.return_type_expr
                    if module.return_type_expr is not None and not specialize_signature
                    else _apply_subst(expr_return_tp, state.ctx)
                ),
            )
            if specialize_signature:
                typed_module = _specialize_definition_body_dim_names(
                    state, module, typed_module, call_expr_defs
                )
            _store_typed_module(state, typed_module)
            return typed_module, _apply_subst(expr_return_tp, state.ctx)
        typed_statements, _, returns = _tc_statements(
            state,
            module.statements,
            base_env,
            dict(call_expr_defs or {}),
            expected_returns,
            protected_dim_names,
            refine_returns_from_body=specialize_signature,
        )
        return_tp = _final_return_type(returns, state.ctx)
        refined_params: list[AxonParam] = []
        for param in module.params:
            param_tp = (signature_env if specialize_signature else base_env).get(
                param.name, param.type_expr
            )
            if param.optional and isinstance(param_tp, TypeOptional):
                param_tp = param_tp.inner
            param_type_expr = (
                param.type_expr
                if param.type_expr is not None and not specialize_signature
                else _apply_subst(param_tp, state.ctx)
                if param_tp is not None
                else None
            )
            if specialize_signature:
                param_type_expr = _preserve_compound_dim_binders(
                    param.type_expr, param_type_expr
                )
            refined_params.append(
                replace(
                    param,
                    type_expr=param_type_expr,
                )
            )
        signature_return_tp = return_tp
        if specialize_signature:
            signature_return_tp = _preserve_compound_dim_binders(
                module.return_type_expr,
                _apply_subst(signature_return_tp, state.ctx)
                if signature_return_tp is not None
                else None,
            )
        return_type_expr = (
            module.return_type_expr
            if module.return_type_expr is not None and not specialize_signature
            else signature_return_tp
        )
        typed_module = replace(
            module,
            params=tuple(refined_params),
            statements=tuple(_normalize_statement(stmt, state.ctx) for stmt in typed_statements),
            return_type_expr=return_type_expr,
        )
        if specialize_signature:
            typed_module = _specialize_definition_body_dim_names(
                state, module, typed_module, call_expr_defs
            )
        _store_typed_module(state, typed_module)
        return typed_module, _apply_subst(return_tp, state.ctx) if return_tp is not None else None
    finally:
        state.active = previous_active


def typecheck2_flat_axon_file(program: AxonFile, *, main_module: str | None = None) -> AxonFile:
    main_module = resolve_main_module(program, main_module=main_module)
    validate_flat_axon_file(program, main_module=main_module)
    program = _prune_to_main(program, main_module)
    modules_by_name = {module.name: module for module in program.modules}

    def new_state() -> _Tc2:
        return _Tc2(
            program=program,
            main_module=main_module,
            modules_by_name=modules_by_name,
            ctx=_TcCtx(
                modules_by_name=modules_by_name,
                type_aliases=dict(program.type_aliases),
                substitutions={},
                dim_substitutions={},
            ),
            typed_modules={},
            specialization_conflicts=set(),
            fresh_dim_names=set(),
            fresh_dim_sources={},
            fresh_type_names=set(),
            fresh_type_sources={},
        )

    demand_state = new_state()
    roots = [main_module] if main_module is not None else [module.name for module in program.modules]
    for root in roots:
        module = modules_by_name[root]
        _tc_definition(demand_state, module)
    typed_by_name: dict[str, AxonDefinition] = {}
    state_by_name: dict[str, _Tc2] = {}
    for module in program.modules:
        typed = demand_state.typed_modules.get(module.name)
        should_emit_generic = (
            module.name not in roots
            and not _is_loop_generated_definition_name(module.name)
        )
        if should_emit_generic or typed is None or (
            module.name in demand_state.specialization_conflicts
            and not _is_loop_generated_definition_name(module.name)
        ):
            generic_state = new_state()
            typed, _ = _tc_definition(generic_state, module)
            state_by_name[module.name] = generic_state
        else:
            state_by_name[module.name] = demand_state
        typed_by_name[module.name] = typed
    _prefer_closed_constant_dim_names(demand_state)
    typed_by_name = _align_loop_continue_signatures(typed_by_name)
    typed_by_name = {
        name: _canonicalize_fresh_module(
            _normalize_typed_module(module, state_by_name[name].ctx),
            state_by_name[name],
        )
        for name, module in typed_by_name.items()
    }
    for root in roots:
        original = modules_by_name[root]
        typed_by_name[root] = replace(
            typed_by_name[root],
            params=original.params,
            return_type_expr=original.return_type_expr,
        )
    typed_modules = tuple(typed_by_name.get(module.name, module) for module in program.modules)
    typed_program = replace(program, modules=typed_modules)
    validate_typed_axon_file(typed_program, main_module=main_module)
    return typed_program
