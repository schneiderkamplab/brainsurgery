from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, cast

from ..ops import get_op_lowering_type_signature, get_op_type_rule
from ..ops._broadcast import broadcast_shape
from .ast import (
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
    ConstraintAtom,
    ConstraintOperand,
    DimExprBinary,
    DimToken,
    TypeAliasDef,
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
    parse_type_expr,
)
from .entrypoint import resolve_main_module
from .resolve import reachable_definitions
from .validate import (
    validate_flat_axon_file,
    validate_normalized_axon_file,
    validate_typed_axon_file,
)

_COMPARE_OPS = {"==", "!=", "<", "<=", ">", ">="}
_BOOL_OPS = {"and", "or"}
_ARITH_OPS = {"+", "-", "*", "/"}


@dataclass
class _TcCtx:
    modules_by_name: dict[str, AxonDefinition]
    type_aliases: dict[str, TypeAliasDef]
    substitutions: dict[str, TypeExpr]
    dim_substitutions: dict[str, DimToken | tuple[DimToken, ...]]
    fresh_counter: int = 0

    def fresh_type_var(self, prefix: str = "__tc") -> TypeVar:
        self.fresh_counter += 1
        return TypeVar(name=f"{prefix}{self.fresh_counter}")

    def child(self, *, share_constraints: bool = False) -> _TcCtx:
        return _TcCtx(
            modules_by_name=self.modules_by_name,
            type_aliases=self.type_aliases,
            substitutions=self.substitutions if share_constraints else {},
            dim_substitutions=self.dim_substitutions if share_constraints else {},
            fresh_counter=self.fresh_counter,
        )


@dataclass(frozen=True)
class _RecursiveInterfaces:
    signatures: dict[str, tuple[list[TypeExpr], list[TypeExpr]]]
    members: frozenset[str]


def _is_generic_named_type(tp: TypeExpr, *, type_aliases: dict[str, TypeAliasDef]) -> bool:
    return (
        isinstance(tp, TypeNamed)
        and tp.name != "Tensor"
        and tp.name not in type_aliases
        and "." not in tp.name
        and "::" not in tp.name
    )


def _apply_subst(tp: TypeExpr, ctx: _TcCtx) -> TypeExpr:
    def _apply_dim_token(dim: DimToken) -> tuple[DimToken, ...]:
        if isinstance(dim, str):
            mapped = ctx.dim_substitutions.get(dim)
            if mapped is None:
                return (dim,)
            if isinstance(mapped, tuple):
                return tuple(_flatten_dim_tokens(mapped))
            return _apply_dim_token(mapped)
        if isinstance(dim, int):
            return (dim,)
        left = _apply_dim_token(dim.left)
        right = _apply_dim_token(dim.right)
        if len(left) == 1 and len(right) == 1:
            return (DimExprBinary(op=dim.op, left=left[0], right=right[0]),)
        return (dim,)

    def _flatten_dim_tokens(dims: tuple[DimToken, ...]) -> tuple[DimToken, ...]:
        out: list[DimToken] = []
        for item in dims:
            out.extend(_apply_dim_token(item))
        return tuple(out)

    if isinstance(tp, TypeVar):
        current: TypeExpr = tp
        seen: set[str] = set()
        while (
            isinstance(current, TypeVar)
            and current.name in ctx.substitutions
            and current.name not in seen
        ):
            seen.add(current.name)
            next_tp = ctx.substitutions[current.name]
            if next_tp == current:
                break
            current = next_tp
        if current != tp:
            return _apply_subst(current, ctx)
        return current
    if isinstance(tp, TypeOptional):
        return TypeOptional(inner=_apply_subst(tp.inner, ctx))
    if isinstance(tp, TypeList):
        return TypeList(item=_apply_subst(tp.item, ctx))
    if isinstance(tp, TypeTuple):
        return TypeTuple(items=tuple(_apply_subst(item, ctx) for item in tp.items))
    if isinstance(tp, TypeTensor):
        return TypeTensor(base=tp.base, dims=_flatten_dim_tokens(tp.dims))
    if isinstance(tp, TypeNamed):
        return TypeNamed(name=tp.name, args=_flatten_dim_tokens(tp.args))
    return tp


def _expand_alias(tp: TypeExpr, ctx: _TcCtx) -> TypeExpr:
    tp = _apply_subst(tp, ctx)
    if isinstance(tp, TypeOptional):
        return TypeOptional(inner=_expand_alias(tp.inner, ctx))
    if isinstance(tp, TypeList):
        return TypeList(item=_expand_alias(tp.item, ctx))
    if isinstance(tp, TypeTuple):
        return TypeTuple(items=tuple(_expand_alias(item, ctx) for item in tp.items))
    if isinstance(tp, TypeNamed):
        alias = ctx.type_aliases.get(tp.name)
        if alias is None and "." in tp.name:
            alias = ctx.type_aliases.get(tp.name.rsplit(".", 1)[1])
        if alias is None:
            return tp
        subst = _match_type_alias_dims(alias.params, tp.args)
        if subst is None:
            return tp
        return _expand_alias(_substitute_type_dims(alias.value, subst=subst), ctx)
    return tp


def _match_type_alias_dims(
    params: tuple[str, ...], args: tuple[DimToken, ...]
) -> dict[str, DimToken | tuple[DimToken, ...]] | None:
    variadic_idx = next((idx for idx, param in enumerate(params) if param.startswith("..")), None)
    if variadic_idx is None:
        if len(args) != len(params):
            return None
        return {name: dim for name, dim in zip(params, args, strict=True)}
    if any(param.startswith("..") for param in params[variadic_idx + 1 :]):
        return None
    fixed_after = len(params) - variadic_idx - 1
    if len(args) < variadic_idx + fixed_after:
        return None
    subst: dict[str, DimToken | tuple[DimToken, ...]] = {}
    for name, dim in zip(params[:variadic_idx], args[:variadic_idx], strict=True):
        subst[name] = dim
    variadic_end = len(args) - fixed_after
    subst[params[variadic_idx]] = tuple(args[variadic_idx:variadic_end])
    if fixed_after:
        for name, dim in zip(
            params[variadic_idx + 1 :], args[variadic_end:], strict=True
        ):
            subst[name] = dim
    return subst


def _optional_inner_type(tp: TypeExpr) -> TypeExpr:
    return tp.inner if isinstance(tp, TypeOptional) else tp


def _is_scalar_numeric_type(tp: TypeExpr) -> bool:
    return isinstance(_optional_inner_type(tp), TypeFloat | TypeInt | TypeDim)


def _substitute_alias_dims(tp: TypeExpr, *, subst: dict[str, DimToken]) -> TypeExpr:
    def _sub_dim(dim: DimToken) -> DimToken:
        if isinstance(dim, str):
            return subst.get(dim, dim)
        if isinstance(dim, int):
            return dim
        return type(dim)(op=dim.op, left=_sub_dim(dim.left), right=_sub_dim(dim.right))

    if isinstance(tp, TypeOptional):
        return TypeOptional(inner=_substitute_alias_dims(tp.inner, subst=subst))
    if isinstance(tp, TypeList):
        return TypeList(item=_substitute_alias_dims(tp.item, subst=subst))
    if isinstance(tp, TypeTuple):
        return TypeTuple(
            items=tuple(_substitute_alias_dims(item, subst=subst) for item in tp.items)
        )
    if isinstance(tp, TypeTensor):
        return TypeTensor(base=tp.base, dims=tuple(_sub_dim(dim) for dim in tp.dims))
    if isinstance(tp, TypeNamed):
        return TypeNamed(name=tp.name, args=tuple(_sub_dim(dim) for dim in tp.args))
    return tp


def _substitute_type_dims(
    tp: TypeExpr, *, subst: Mapping[str, DimToken | tuple[DimToken, ...]]
) -> TypeExpr:
    def _sub_dim(dim: DimToken) -> tuple[DimToken, ...]:
        if isinstance(dim, str):
            mapped = subst.get(dim)
            if mapped is None:
                return (dim,)
            if isinstance(mapped, tuple):
                return mapped
            return (mapped,)
        if isinstance(dim, int):
            return (dim,)
        left = _sub_dim(dim.left)
        right = _sub_dim(dim.right)
        if len(left) == 1 and len(right) == 1:
            return (DimExprBinary(op=dim.op, left=left[0], right=right[0]),)
        return (dim,)

    if isinstance(tp, TypeOptional):
        return TypeOptional(inner=_substitute_type_dims(tp.inner, subst=subst))
    if isinstance(tp, TypeList):
        return TypeList(item=_substitute_type_dims(tp.item, subst=subst))
    if isinstance(tp, TypeTuple):
        return TypeTuple(items=tuple(_substitute_type_dims(item, subst=subst) for item in tp.items))
    if isinstance(tp, TypeTensor):
        dims: list[DimToken] = []
        for dim in tp.dims:
            dims.extend(_sub_dim(dim))
        return TypeTensor(base=tp.base, dims=tuple(dims))
    if isinstance(tp, TypeNamed):
        args: list[DimToken] = []
        for dim in tp.args:
            args.extend(_sub_dim(dim))
        return TypeNamed(name=tp.name, args=tuple(args))
    return tp


def _has_variadic_dims(dims: tuple[DimToken, ...]) -> bool:
    return any(isinstance(dim, str) and dim.startswith("..") for dim in dims)


def _occurs(var_name: str, tp: TypeExpr, ctx: _TcCtx) -> bool:
    tp = _apply_subst(tp, ctx)
    if isinstance(tp, TypeVar):
        return tp.name == var_name
    if isinstance(tp, TypeOptional):
        return _occurs(var_name, tp.inner, ctx)
    if isinstance(tp, TypeList):
        return _occurs(var_name, tp.item, ctx)
    if isinstance(tp, TypeTuple):
        return any(_occurs(var_name, item, ctx) for item in tp.items)
    return False


def _bind_var(var: TypeVar, tp: TypeExpr, ctx: _TcCtx) -> None:
    tp = _apply_subst(tp, ctx)
    if isinstance(tp, TypeVar) and tp.name == var.name:
        return
    if _occurs(var.name, tp, ctx):
        raise ValueError(f"Axon typecheck failed: recursive type variable {var.name!r}")
    ctx.substitutions[var.name] = tp


def _dim_name_priority(name: str) -> tuple[int, int]:
    def _priority(inner: str, full_len: int) -> tuple[int, int]:
        if (
            inner.startswith("__d")
            or inner.startswith("__flat_")
            or inner.startswith("__gdim")
            or inner.startswith("__tc")
            or inner.startswith("_T")
        ):
            return (0, -full_len)
        if len(inner) == 1:
            if any(ch.isupper() for ch in inner):
                return (2, -full_len)
            if any(ch.islower() for ch in inner):
                return (1, -full_len)
        if any(ch.isupper() for ch in inner):
            return (4, -full_len)
        if any(ch.islower() for ch in inner):
            return (3, -full_len)
        return (3, -full_len)

    if name.startswith(".."):
        return _priority(name[2:], len(name))
    return _priority(name, len(name))


def _typevar_name_priority(name: str) -> tuple[int, int]:
    if name.startswith("__tc") or name.startswith("_T"):
        return (0, -len(name))
    if "::" in name:
        return (2, -len(name))
    return (1, -len(name))


def _apply_dim_subst_token(dim: DimToken, ctx: _TcCtx) -> DimToken | tuple[DimToken, ...]:
    if isinstance(dim, str):
        mapped = ctx.dim_substitutions.get(dim)
        if mapped is None:
            return dim
        if isinstance(mapped, tuple):
            return tuple(_normalize_dim_token(item, ctx) for item in mapped)
        return _normalize_dim_token(mapped, ctx)
    if isinstance(dim, int):
        return dim
    return DimExprBinary(
        op=dim.op,
        left=_normalize_dim_token(dim.left, ctx),
        right=_normalize_dim_token(dim.right, ctx),
    )


def _normalize_dim_token(dim: DimToken, ctx: _TcCtx) -> DimToken:
    normalized = _apply_dim_subst_token(dim, ctx)
    if isinstance(normalized, tuple):
        if len(normalized) == 1:
            return normalized[0]
        return dim
    if isinstance(normalized, DimExprBinary):
        return _simplify_dim_expr(normalized)
    return normalized


def _simplify_dim_expr(dim: DimExprBinary) -> DimToken:
    left: DimToken = _simplify_dim_expr(dim.left) if isinstance(dim.left, DimExprBinary) else dim.left
    right: DimToken = _simplify_dim_expr(dim.right) if isinstance(dim.right, DimExprBinary) else dim.right
    if isinstance(left, int) and isinstance(right, int):
        if dim.op == "+":
            return left + right
        if dim.op == "-":
            return left - right
        if dim.op == "*":
            return left * right
        if dim.op == "/" and right != 0 and left % right == 0:
            return left // right
    if dim.op == "+":
        if right == 0:
            return left
        if left == 0:
            return right
    if dim.op == "-":
        if right == 0:
            return left
        if left == right:
            return 0
        if isinstance(right, DimExprBinary) and right.op == "-" and right.left == left:
            return right.right
        if isinstance(left, DimExprBinary) and left.op == "+" and left.right == right:
            return left.left
        if isinstance(left, DimExprBinary) and left.op == "+" and left.left == right:
            return left.right
    if dim.op == "*":
        if isinstance(right, DimExprBinary) and right.op == "/" and right.right == left:
            return right.left
        if isinstance(left, DimExprBinary) and left.op == "/" and left.right == right:
            return left.left
        if right == 1:
            return left
        if left == 1:
            return right
        if right == 0 or left == 0:
            return 0
    if dim.op == "/" and right == 1:
        return left
    if dim.op == "/":
        if isinstance(left, DimExprBinary) and left.op == "*":
            if left.left == right:
                return left.right
            if left.right == right:
                return left.left
    return DimExprBinary(op=dim.op, left=left, right=right)


def _dim_value_contains_name(value: DimToken | tuple[DimToken, ...], name: str) -> bool:
    if isinstance(value, tuple):
        return any(_dim_value_contains_name(item, name) for item in value)
    if value == name:
        return True
    if isinstance(value, DimExprBinary):
        return name in dim_token_names(value)
    return False


def _bind_dim_name(name: str, value: DimToken | tuple[DimToken, ...], ctx: _TcCtx) -> None:
    if _dim_value_contains_name(value, name):
        if value == name or (isinstance(value, tuple) and len(value) == 1 and value[0] == name):
            return
        raise ValueError(f"Axon typecheck failed: recursive dim variable {name!r}")
    if name.startswith(".."):
        ctx.dim_substitutions[name] = value
        return
    if isinstance(value, tuple):
        if len(value) != 1:
            ctx.dim_substitutions[name] = value
            return
        value = value[0]
    value = _normalize_dim_token(value, ctx)
    if value == name:
        return
    if name in dim_token_names(value):
        raise ValueError(f"Axon typecheck failed: recursive dim variable {name!r}")
    ctx.dim_substitutions[name] = value


def _unify_dim_token(left: DimToken, right: DimToken, ctx: _TcCtx) -> DimToken:
    left = _normalize_dim_token(left, ctx)
    right = _normalize_dim_token(right, ctx)
    if left == right:
        return left
    if (
        isinstance(left, str)
        and not left.startswith("..")
        and isinstance(right, str)
        and not right.startswith("..")
    ):
        if _dim_name_priority(left) >= _dim_name_priority(right):
            _bind_dim_name(right, left, ctx)
            return _normalize_dim_token(left, ctx)
        _bind_dim_name(left, right, ctx)
        return _normalize_dim_token(right, ctx)
    if isinstance(left, str) and not left.startswith(".."):
        _bind_dim_name(left, right, ctx)
        return _normalize_dim_token(right, ctx)
    if isinstance(right, str) and not right.startswith(".."):
        _bind_dim_name(right, left, ctx)
        return _normalize_dim_token(left, ctx)
    if isinstance(left, int) and isinstance(right, int):
        if left != right:
            raise ValueError(f"Axon typecheck failed: dim mismatch {left!r} vs {right!r}")
        return left
    if isinstance(left, DimExprBinary) and isinstance(right, DimExprBinary) and left.op == right.op:
        return DimExprBinary(
            op=left.op,
            left=_unify_dim_token(left.left, right.left, ctx),
            right=_unify_dim_token(left.right, right.right, ctx),
        )
    raise ValueError(f"Axon typecheck failed: dim mismatch {left!r} vs {right!r}")


def _unify_dim_sequence(
    expected: tuple[DimToken, ...],
    actual: tuple[DimToken, ...],
    ctx: _TcCtx,
) -> tuple[DimToken, ...]:
    def _unify_fixed_pairs(pairs: list[tuple[DimToken, DimToken]]) -> tuple[DimToken, ...]:
        out: list[DimToken | None] = []
        deferred: list[tuple[int, DimToken, DimToken, ValueError]] = []
        for left, right in pairs:
            try:
                out.append(_unify_dim_token(left, right, ctx))
            except ValueError as exc:
                if "recursive dim variable" not in str(exc):
                    raise
                deferred.append((len(out), left, right, exc))
                out.append(None)
        for idx, left, right, original_exc in deferred:
            try:
                out[idx] = _unify_dim_token(left, right, ctx)
            except ValueError as exc:
                if "recursive dim variable" in str(exc):
                    raise original_exc
                raise
        return tuple(cast(DimToken, item) for item in out)

    left_var = next(
        (idx for idx, dim in enumerate(expected) if isinstance(dim, str) and dim.startswith("..")),
        None,
    )
    right_var = next(
        (idx for idx, dim in enumerate(actual) if isinstance(dim, str) and dim.startswith("..")),
        None,
    )
    if left_var is None and right_var is None:
        if len(expected) != len(actual):
            raise ValueError("Axon typecheck failed: tensor rank mismatch")
        return _unify_fixed_pairs(list(zip(expected, actual, strict=True)))
    if left_var is not None and right_var is None:
        prefix = expected[:left_var]
        suffix = expected[left_var + 1 :]
        if len(actual) < len(prefix) + len(suffix):
            raise ValueError("Axon typecheck failed: tensor rank mismatch")
        head = [
            _unify_dim_token(left, right, ctx)
            for left, right in zip(prefix, actual[: len(prefix)], strict=True)
        ]
        tail = [
            _unify_dim_token(left, right, ctx)
            for left, right in zip(suffix, actual[-len(suffix) :] if suffix else (), strict=True)
        ]
        variadic_name = expected[left_var]
        assert isinstance(variadic_name, str)
        middle = actual[len(prefix) : len(actual) - len(suffix) if suffix else len(actual)]
        _bind_dim_name(variadic_name, middle, ctx)
        return tuple(head) + tuple(middle) + tuple(tail)
    if right_var is not None and left_var is None:
        return _unify_dim_sequence(actual, expected, ctx)
    assert left_var is not None and right_var is not None
    left_prefix = expected[:left_var]
    left_suffix = expected[left_var + 1 :]
    right_prefix = actual[:right_var]
    right_suffix = actual[right_var + 1 :]
    if not left_prefix and not left_suffix:
        left_name = expected[left_var]
        assert isinstance(left_name, str)
        _bind_dim_name(left_name, actual, ctx)
        return tuple(_normalize_dim_token(dim, ctx) for dim in actual)
    if not right_prefix and not right_suffix:
        right_name = actual[right_var]
        assert isinstance(right_name, str)
        _bind_dim_name(right_name, expected, ctx)
        return tuple(_normalize_dim_token(dim, ctx) for dim in expected)
    if not left_prefix and left_suffix and len(actual) >= len(left_suffix):
        tail_actual = actual[-len(left_suffix) :]
        tail = [
            _unify_dim_token(left, right, ctx)
            for left, right in zip(left_suffix, tail_actual, strict=True)
        ]
        left_name = expected[left_var]
        assert isinstance(left_name, str)
        middle = actual[: len(actual) - len(left_suffix)]
        _bind_dim_name(left_name, middle, ctx)
        return tuple(_normalize_dim_token(dim, ctx) for dim in middle) + tuple(tail)
    if not right_prefix and right_suffix and len(expected) >= len(right_suffix):
        tail_expected = expected[-len(right_suffix) :]
        tail = [
            _unify_dim_token(left, right, ctx)
            for left, right in zip(tail_expected, right_suffix, strict=True)
        ]
        right_name = actual[right_var]
        assert isinstance(right_name, str)
        middle = expected[: len(expected) - len(right_suffix)]
        _bind_dim_name(right_name, middle, ctx)
        return tuple(_normalize_dim_token(dim, ctx) for dim in middle) + tuple(tail)
    if len(left_prefix) != len(right_prefix) or len(left_suffix) != len(right_suffix):
        raise ValueError("Axon typecheck failed: cannot unify multiple variadic tensor dims")
    head = [
        _unify_dim_token(left, right, ctx)
        for left, right in zip(left_prefix, right_prefix, strict=True)
    ]
    tail = [
        _unify_dim_token(left, right, ctx)
        for left, right in zip(left_suffix, right_suffix, strict=True)
    ]
    left_name = expected[left_var]
    right_name = actual[right_var]
    assert isinstance(left_name, str)
    assert isinstance(right_name, str)
    left_middle = expected[
        len(left_prefix) : len(expected) - len(left_suffix) if left_suffix else len(expected)
    ]
    right_middle = actual[
        len(right_prefix) : len(actual) - len(right_suffix) if right_suffix else len(actual)
    ]
    if (
        len(left_middle) == 1
        and left_middle[0] == left_name
        and len(right_middle) == 1
        and right_middle[0] == right_name
    ):
        if _dim_name_priority(left_name) >= _dim_name_priority(right_name):
            _bind_dim_name(right_name, left_middle, ctx)
            middle = tuple(_normalize_dim_token(dim, ctx) for dim in left_middle)
        else:
            _bind_dim_name(left_name, right_middle, ctx)
            middle = tuple(_normalize_dim_token(dim, ctx) for dim in right_middle)
    else:
        if len(left_middle) == 1 and left_middle[0] == left_name:
            _bind_dim_name(left_name, right_middle, ctx)
        middle = tuple(_normalize_dim_token(dim, ctx) for dim in right_middle)
    return tuple(head) + middle + tuple(tail)


def _unify(left: TypeExpr, right: TypeExpr, ctx: _TcCtx) -> TypeExpr:
    left = _expand_alias(_apply_subst(left, ctx), ctx)
    right = _expand_alias(_apply_subst(right, ctx), ctx)
    if left == right:
        return left
    if isinstance(left, TypeVar) and isinstance(right, TypeVar):
        if _typevar_name_priority(left.name) >= _typevar_name_priority(right.name):
            _bind_var(right, left, ctx)
            return _apply_subst(left, ctx)
        _bind_var(left, right, ctx)
        return _apply_subst(right, ctx)
    if isinstance(left, TypeVar):
        _bind_var(left, right, ctx)
        return _apply_subst(right, ctx)
    if isinstance(right, TypeVar):
        _bind_var(right, left, ctx)
        return _apply_subst(left, ctx)
    if isinstance(left, TypeAny):
        return right
    if isinstance(right, TypeAny):
        return left
    if (isinstance(left, TypeInt) and isinstance(right, TypeFloat)) or (
        isinstance(left, TypeFloat) and isinstance(right, TypeInt)
    ):
        return TypeFloat()
    if (isinstance(left, TypeDim) and isinstance(right, TypeFloat)) or (
        isinstance(left, TypeFloat) and isinstance(right, TypeDim)
    ):
        return TypeFloat()
    if (isinstance(left, TypeInt) and isinstance(right, TypeDim)) or (
        isinstance(left, TypeDim) and isinstance(right, TypeInt)
    ):
        return TypeDim()
    if isinstance(left, TypeOptional) and isinstance(right, TypeNull):
        return left
    if isinstance(right, TypeOptional) and isinstance(left, TypeNull):
        return right
    if isinstance(left, TypeNull):
        return TypeOptional(inner=right)
    if isinstance(right, TypeNull):
        return TypeOptional(inner=left)
    if isinstance(left, TypeOptional) and isinstance(right, TypeOptional):
        return TypeOptional(inner=_unify(left.inner, right.inner, ctx))
    if isinstance(left, TypeOptional):
        return TypeOptional(inner=_unify(left.inner, right, ctx))
    if isinstance(right, TypeOptional):
        return TypeOptional(inner=_unify(left, right.inner, ctx))
    if isinstance(left, TypeList) and isinstance(right, TypeList):
        return TypeList(item=_unify(left.item, right.item, ctx))
    if isinstance(left, TypeTuple) and isinstance(right, TypeTuple):
        if len(left.items) != len(right.items):
            raise ValueError("Axon typecheck failed: tuple arity mismatch")
        return TypeTuple(
            items=tuple(_unify(a, b, ctx) for a, b in zip(left.items, right.items, strict=True))
        )
    if isinstance(left, TypeTensor) and isinstance(right, TypeTensor):
        if left.base != right.base and left.base != "Tensor" and right.base != "Tensor":
            raise ValueError(
                f"Axon typecheck failed: tensor base mismatch {left.base!r} vs {right.base!r}"
            )
        if left.dims and right.dims:
            return TypeTensor(base="Tensor", dims=_unify_dim_sequence(left.dims, right.dims, ctx))
        if left.dims:
            return TypeTensor(
                base="Tensor",
                dims=tuple(_normalize_dim_token(dim, ctx) for dim in left.dims),
            )
        return TypeTensor(
            base="Tensor",
            dims=tuple(_normalize_dim_token(dim, ctx) for dim in right.dims),
        )
    if type(left) is type(right):
        return left
    if (
        isinstance(left, TypeNamed)
        and isinstance(right, TypeNamed)
        and left.name == right.name
        and left.args == right.args
    ):
        return left
    raise ValueError(f"Axon typecheck failed: cannot unify {left!r} with {right!r}")


def _broadcast_tensor_branch_types(
    left: TypeExpr,
    right: TypeExpr,
    ctx: _TcCtx,
) -> TypeExpr:
    left_applied = _apply_subst(left, ctx)
    right_applied = _apply_subst(right, ctx)
    left_expanded = _expand_alias(left_applied, ctx)
    right_expanded = _expand_alias(right_applied, ctx)
    if not isinstance(left_expanded, TypeTensor) or not isinstance(right_expanded, TypeTensor):
        return _unify(left, right, ctx)
    if (
        left_expanded.base != right_expanded.base
        and left_expanded.base != "Tensor"
        and right_expanded.base != "Tensor"
    ):
        return _unify(left, right, ctx)
    if not left_expanded.dims or not right_expanded.dims:
        return _unify(left, right, ctx)
    dims = broadcast_shape(
        tuple(_normalize_dim_token(dim, ctx) for dim in left_expanded.dims),
        tuple(_normalize_dim_token(dim, ctx) for dim in right_expanded.dims),
    )
    if dims is None:
        return _unify(left, right, ctx)
    return TypeTensor(base="Tensor", dims=tuple(_normalize_dim_token(dim, ctx) for dim in dims))


def _fresh_dim_name(ctx: _TcCtx) -> str:
    ctx.fresh_counter += 1
    return f"__d{ctx.fresh_counter}"


def _join_dim_token(left: DimToken, right: DimToken, ctx: _TcCtx) -> DimToken:
    left = _normalize_dim_token(left, ctx)
    right = _normalize_dim_token(right, ctx)
    if left == right:
        return left
    try:
        return _unify_dim_token(left, right, ctx)
    except ValueError:
        return _fresh_dim_name(ctx)


def _join_tensor_dims(
    left: tuple[DimToken, ...],
    right: tuple[DimToken, ...],
    ctx: _TcCtx,
) -> tuple[DimToken, ...] | None:
    if len(left) != len(right):
        return None
    return tuple(
        _join_dim_token(left_dim, right_dim, ctx)
        for left_dim, right_dim in zip(left, right, strict=True)
    )


def _unify_broadcast_tensor_dims(
    left: tuple[DimToken, ...],
    right: tuple[DimToken, ...],
    ctx: _TcCtx,
) -> tuple[DimToken, ...] | None:
    max_rank = max(len(left), len(right))
    left_full: tuple[DimToken, ...] = (1,) * (max_rank - len(left)) + left
    right_full: tuple[DimToken, ...] = (1,) * (max_rank - len(right)) + right
    merged: list[DimToken] = []
    for left_dim, right_dim in zip(left_full, right_full, strict=True):
        left_dim = _normalize_dim_token(left_dim, ctx)
        right_dim = _normalize_dim_token(right_dim, ctx)
        if left_dim == right_dim:
            merged.append(left_dim)
            continue
        if left_dim == 1:
            merged.append(right_dim)
            continue
        if right_dim == 1:
            merged.append(left_dim)
            continue
        try:
            merged.append(_unify_dim_token(left_dim, right_dim, ctx))
        except ValueError:
            return None
    return tuple(_normalize_dim_token(dim, ctx) for dim in merged)


def _join_branch_types(left: TypeExpr, right: TypeExpr, ctx: _TcCtx) -> TypeExpr:
    left_applied = _apply_subst(left, ctx)
    right_applied = _apply_subst(right, ctx)
    left_expanded = _expand_alias(left_applied, ctx)
    right_expanded = _expand_alias(right_applied, ctx)
    if isinstance(left_expanded, TypeTuple) and isinstance(right_expanded, TypeTuple):
        if len(left_expanded.items) != len(right_expanded.items):
            return _unify(left, right, ctx)
        return TypeTuple(
            items=tuple(
                _join_branch_types(left_item, right_item, ctx)
                for left_item, right_item in zip(
                    left_expanded.items, right_expanded.items, strict=True
                )
            )
        )
    if isinstance(left_expanded, TypeOptional) and isinstance(right_expanded, TypeOptional):
        return TypeOptional(inner=_join_branch_types(left_expanded.inner, right_expanded.inner, ctx))
    if isinstance(left_expanded, TypeOptional):
        return TypeOptional(inner=_join_branch_types(left_expanded.inner, right_expanded, ctx))
    if isinstance(right_expanded, TypeOptional):
        return TypeOptional(inner=_join_branch_types(left_expanded, right_expanded.inner, ctx))
    if isinstance(left_expanded, TypeTensor) and isinstance(right_expanded, TypeTensor):
        if (
            left_expanded.base != right_expanded.base
            and left_expanded.base != "Tensor"
            and right_expanded.base != "Tensor"
        ):
            return _unify(left, right, ctx)
        if left_expanded.dims and right_expanded.dims:
            joined_dims = _join_tensor_dims(left_expanded.dims, right_expanded.dims, ctx)
            if joined_dims is not None:
                return TypeTensor(base="Tensor", dims=joined_dims)
        return _broadcast_tensor_branch_types(left, right, ctx)
    return _unify(left, right, ctx)


def _scoped_typevars(
    tp: TypeExpr | None,
    *,
    module_name: str,
    ctx: _TcCtx,
    freshen_generics: bool,
    preserve_dim_names: set[str] | None = None,
) -> TypeExpr:
    preserved = preserve_dim_names or set()
    dim_subst: dict[str, DimToken] = {}

    def _fresh_dim_name(name: str) -> str:
        existing = dim_subst.get(name)
        if isinstance(existing, str):
            return existing
        ctx.fresh_counter += 1
        fresh = f"__d{ctx.fresh_counter}"
        if name.startswith(".."):
            fresh = ".." + fresh
        dim_subst[name] = fresh
        return fresh

    def _rewrite_dim(dim: DimToken) -> DimToken:
        if isinstance(dim, str):
            if dim in preserved:
                return dim
            return _fresh_dim_name(dim) if freshen_generics else dim
        if isinstance(dim, int):
            return dim
        return DimExprBinary(op=dim.op, left=_rewrite_dim(dim.left), right=_rewrite_dim(dim.right))

    def _rewrite_type(inner: TypeExpr) -> TypeExpr:
        if isinstance(inner, TypeVar):
            if not freshen_generics and "::" in inner.name:
                return inner
            if freshen_generics:
                return ctx.fresh_type_var()
            return TypeVar(name=f"{module_name}::{inner.name}")
        if isinstance(inner, TypeNamed) and _is_generic_named_type(
            inner, type_aliases=ctx.type_aliases
        ):
            if freshen_generics:
                return ctx.fresh_type_var()
            return TypeVar(name=f"{module_name}::{inner.name}")
        if isinstance(inner, TypeOptional):
            return TypeOptional(inner=_rewrite_type(inner.inner))
        if isinstance(inner, TypeList):
            return TypeList(item=_rewrite_type(inner.item))
        if isinstance(inner, TypeTuple):
            return TypeTuple(items=tuple(_rewrite_type(item) for item in inner.items))
        if isinstance(inner, TypeNamed):
            return TypeNamed(name=inner.name, args=tuple(_rewrite_dim(dim) for dim in inner.args))
        if isinstance(inner, TypeTensor):
            return TypeTensor(base=inner.base, dims=tuple(_rewrite_dim(dim) for dim in inner.dims))
        return inner

    if tp is None:
        return TypeAny()
    return _rewrite_type(tp)


def _module_return_types(
    module: AxonDefinition, ctx: _TcCtx, *, freshen_generics: bool
) -> tuple[TypeExpr, ...]:
    preserve_dim_names = {
        param.name
        for param in module.params
        if isinstance(param.type_expr, TypeDim)
        or (
            isinstance(param.type_expr, TypeOptional) and isinstance(param.type_expr.inner, TypeDim)
        )
    }
    if module.return_type_expr is None:
        return (TypeAny(),)
    scoped = _scoped_typevars(
        module.return_type_expr,
        module_name=module.name,
        ctx=ctx,
        freshen_generics=freshen_generics,
        preserve_dim_names=preserve_dim_names,
    )
    if isinstance(scoped, TypeTuple):
        return scoped.items
    return (scoped,)


def _instantiate_module_signature(
    module: AxonDefinition, ctx: _TcCtx
) -> tuple[list[TypeExpr], list[TypeExpr]]:
    preserve_dim_names: set[str] = set()
    if _is_generated_helper(module.name):
        for param in module.params:
            preserve_dim_names.update(
                name
                for name in _collect_type_dim_names(param.type_expr)
                if not _is_generated_dim_name(name)
            )
        preserve_dim_names.update(
            name
            for name in _collect_type_dim_names(module.return_type_expr)
            if not _is_generated_dim_name(name)
        )
    dim_subst: dict[str, DimToken] = {}
    type_subst: dict[str, TypeVar] = {}

    def _fresh_dim_name(name: str) -> str:
        existing = dim_subst.get(name)
        if isinstance(existing, str):
            return existing
        ctx.fresh_counter += 1
        fresh = f"__d{ctx.fresh_counter}"
        if name.startswith(".."):
            fresh = ".." + fresh
        dim_subst[name] = fresh
        return fresh

    def _rewrite_dim(dim: DimToken) -> DimToken:
        if isinstance(dim, str):
            if dim in preserve_dim_names:
                return dim
            return _fresh_dim_name(dim)
        if isinstance(dim, int):
            return dim
        return DimExprBinary(op=dim.op, left=_rewrite_dim(dim.left), right=_rewrite_dim(dim.right))

    def _rewrite_type(tp: TypeExpr | None) -> TypeExpr:
        def _generic_key(name: str) -> str:
            return name.rsplit("::", 1)[-1]

        if tp is None:
            return TypeAny()
        if isinstance(tp, TypeVar):
            key = _generic_key(tp.name)
            existing = type_subst.get(key)
            if existing is None:
                existing = ctx.fresh_type_var()
                type_subst[key] = existing
            return existing
        if isinstance(tp, TypeNamed) and _is_generic_named_type(tp, type_aliases=ctx.type_aliases):
            key = _generic_key(tp.name)
            existing = type_subst.get(key)
            if existing is None:
                existing = ctx.fresh_type_var()
                type_subst[key] = existing
            return existing
        if isinstance(tp, TypeOptional):
            return TypeOptional(inner=_rewrite_type(tp.inner))
        if isinstance(tp, TypeList):
            return TypeList(item=_rewrite_type(tp.item))
        if isinstance(tp, TypeTuple):
            return TypeTuple(items=tuple(_rewrite_type(item) for item in tp.items))
        if isinstance(tp, TypeNamed):
            return TypeNamed(name=tp.name, args=tuple(_rewrite_dim(dim) for dim in tp.args))
        if isinstance(tp, TypeTensor):
            return TypeTensor(base=tp.base, dims=tuple(_rewrite_dim(dim) for dim in tp.dims))
        return tp

    param_types: list[TypeExpr] = [TypePath() for _ in module.path_params]
    for param in module.params:
        param_type = _rewrite_type(param.type_expr)
        if param.optional:
            param_type = TypeOptional(inner=param_type)
        param_types.append(param_type)
    if module.return_type_expr is None:
        return param_types, [TypeAny()]
    return_tp = _rewrite_type(module.return_type_expr)
    if isinstance(return_tp, TypeTuple):
        return param_types, list(return_tp.items)
    return param_types, [return_tp]


def _clone_type_expr(tp: TypeExpr) -> TypeExpr:
    if isinstance(tp, TypeOptional):
        return TypeOptional(inner=_clone_type_expr(tp.inner))
    if isinstance(tp, TypeList):
        return TypeList(item=_clone_type_expr(tp.item))
    if isinstance(tp, TypeTuple):
        return TypeTuple(items=tuple(_clone_type_expr(item) for item in tp.items))
    if isinstance(tp, TypeTensor):
        return TypeTensor(base=tp.base, dims=tuple(tp.dims))
    if isinstance(tp, TypeNamed):
        return TypeNamed(name=tp.name, args=tuple(tp.args))
    if isinstance(tp, TypeVar):
        return TypeVar(name=tp.name)
    return tp


def _clone_signature(
    signature: tuple[list[TypeExpr], list[TypeExpr]],
) -> tuple[list[TypeExpr], list[TypeExpr]]:
    param_types, return_types = signature
    return (
        [_clone_type_expr(tp) for tp in param_types],
        [_clone_type_expr(tp) for tp in return_types],
    )


def _signature_from_module_header(module: AxonDefinition) -> tuple[list[TypeExpr], list[TypeExpr]]:
    param_types: list[TypeExpr] = [TypePath() for _ in module.path_params]
    for param in module.params:
        param_tp = _clone_type_expr(param.type_expr or TypeAny())
        if param.optional:
            param_tp = TypeOptional(inner=param_tp)
        param_types.append(param_tp)
    if module.return_type_expr is None:
        return param_types, [TypeAny()]
    return_tp = _clone_type_expr(module.return_type_expr)
    if isinstance(return_tp, TypeTuple):
        return param_types, [_clone_type_expr(item) for item in return_tp.items]
    return param_types, [return_tp]


def _canonicalize_generated_signature_type(
    tp: TypeExpr,
    *,
    type_names: dict[str, str],
    dim_names: dict[str, str],
    type_counter: list[int],
    dim_counter: list[int],
) -> TypeExpr:
    def _canon_dim(dim: DimToken) -> DimToken:
        if isinstance(dim, str):
            if _is_generated_dim_name(dim):
                variadic = dim.startswith("..")
                mapped = dim_names.get(dim)
                if mapped is None:
                    dim_counter[0] += 1
                    mapped = f"__gdim{dim_counter[0]}"
                    dim_names[dim] = mapped
                if variadic:
                    return f"..{mapped}"
                return mapped
            return dim
        if isinstance(dim, int):
            return dim
        return DimExprBinary(op=dim.op, left=_canon_dim(dim.left), right=_canon_dim(dim.right))

    if isinstance(tp, TypeOptional):
        return TypeOptional(
            inner=_canonicalize_generated_signature_type(
                tp.inner,
                type_names=type_names,
                dim_names=dim_names,
                type_counter=type_counter,
                dim_counter=dim_counter,
            )
        )
    if isinstance(tp, TypeList):
        return TypeList(
            item=_canonicalize_generated_signature_type(
                tp.item,
                type_names=type_names,
                dim_names=dim_names,
                type_counter=type_counter,
                dim_counter=dim_counter,
            )
        )
    if isinstance(tp, TypeTuple):
        return TypeTuple(
            items=tuple(
                _canonicalize_generated_signature_type(
                    item,
                    type_names=type_names,
                    dim_names=dim_names,
                    type_counter=type_counter,
                    dim_counter=dim_counter,
                )
                for item in tp.items
            )
        )
    if isinstance(tp, TypeTensor):
        return TypeTensor(base=tp.base, dims=tuple(_canon_dim(dim) for dim in tp.dims))
    if isinstance(tp, TypeNamed):
        return TypeNamed(name=tp.name, args=tuple(_canon_dim(dim) for dim in tp.args))
    if isinstance(tp, TypeVar):
        if _is_generated_dim_name(tp.name) or "::" in tp.name:
            mapped = type_names.get(tp.name)
            if mapped is None:
                type_counter[0] += 1
                mapped = f"__gt{type_counter[0]}"
                type_names[tp.name] = mapped
            return TypeVar(name=mapped)
        return TypeVar(name=tp.name)
    return _clone_type_expr(tp)


def _canonicalize_generated_signature(
    signature: tuple[list[TypeExpr], list[TypeExpr]],
) -> tuple[list[TypeExpr], list[TypeExpr]]:
    type_names: dict[str, str] = {}
    dim_names: dict[str, str] = {}
    type_counter = [0]
    dim_counter = [0]
    return (
        [
            _canonicalize_generated_signature_type(
                tp,
                type_names=type_names,
                dim_names=dim_names,
                type_counter=type_counter,
                dim_counter=dim_counter,
            )
            for tp in signature[0]
        ],
        [
            _canonicalize_generated_signature_type(
                tp,
                type_names=type_names,
                dim_names=dim_names,
                type_counter=type_counter,
                dim_counter=dim_counter,
            )
            for tp in signature[1]
        ],
    )


def _module_param_types(
    module: AxonDefinition, ctx: _TcCtx, *, freshen_generics: bool
) -> list[TypeExpr]:
    preserve_dim_names = {
        param.name
        for param in module.params
        if isinstance(param.type_expr, TypeDim)
        or (
            isinstance(param.type_expr, TypeOptional) and isinstance(param.type_expr.inner, TypeDim)
        )
    }
    out: list[TypeExpr] = [TypePath() for _ in module.path_params]
    for param in module.params:
        param_type = _scoped_typevars(
            param.type_expr,
            module_name=module.name,
            ctx=ctx,
            freshen_generics=freshen_generics,
            preserve_dim_names=preserve_dim_names,
        )
        if param.optional:
            param_type = TypeOptional(inner=param_type)
        out.append(param_type)
    return out


def _type_dims(tp: TypeExpr, ctx: _TcCtx) -> tuple[DimToken, ...] | None:
    expanded = _expand_alias(tp, ctx)
    if isinstance(expanded, TypeOptional):
        return _type_dims(expanded.inner, ctx)
    if isinstance(expanded, TypeTensor):
        return expanded.dims
    return None


def _expr_to_dim_token(expr: AxonExpr) -> DimToken | None:
    if isinstance(expr, AxonExprAscribe):
        return _expr_to_dim_token(expr.expr)
    if isinstance(expr, AxonExprParen):
        return _expr_to_dim_token(expr.inner)
    if isinstance(expr, AxonExprInt):
        return expr.value
    if isinstance(expr, AxonExprName):
        return expr.name
    if isinstance(expr, AxonExprCall) and not expr.args and not expr.kwargs:
        return expr.callee
    if isinstance(expr, AxonExprBinary) and expr.op in {"+", "-", "*", "/"}:
        left = _expr_to_dim_token(expr.left)
        right = _expr_to_dim_token(expr.right)
        if left is None or right is None:
            return None
        return DimExprBinary(op=expr.op, left=left, right=right)
    return None


def _expr_to_dim_token_resolved(
    expr: AxonExpr, expr_defs: dict[str, AxonExpr], seen: frozenset[str] = frozenset()
) -> DimToken | None:
    def _resolve_token(token: DimToken, seen_names: frozenset[str]) -> DimToken:
        if isinstance(token, str):
            if token in seen_names:
                return token
            resolved = expr_defs.get(token)
            if resolved is None:
                return token
            resolved_token = _expr_to_dim_token_resolved(
                resolved, expr_defs, seen_names | {token}
            )
            return resolved_token if resolved_token is not None else token
        if isinstance(token, int):
            return token
        return DimExprBinary(
            op=token.op,
            left=_resolve_token(token.left, seen_names),
            right=_resolve_token(token.right, seen_names),
        )

    token = _expr_to_dim_token(expr)
    if token is None:
        return None
    return _resolve_token(token, seen)


def _resolve_dim_alias_token(
    dim: DimToken,
    expr_defs: dict[str, AxonExpr],
    seen: frozenset[str] = frozenset(),
) -> DimToken:
    if isinstance(dim, str):
        if dim in seen:
            return dim
        resolved_expr = expr_defs.get(dim)
        if resolved_expr is None:
            return dim
        resolved = _expr_to_dim_token_resolved(resolved_expr, expr_defs, seen | {dim})
        if resolved is None or resolved == dim:
            return dim
        return _resolve_dim_alias_token(resolved, expr_defs, seen | {dim})
    if isinstance(dim, int):
        return dim
    return _simplify_dim_expr(
        DimExprBinary(
            op=dim.op,
            left=_resolve_dim_alias_token(dim.left, expr_defs, seen),
            right=_resolve_dim_alias_token(dim.right, expr_defs, seen),
        )
    )


def _resolve_type_dim_aliases(tp: TypeExpr, expr_defs: dict[str, AxonExpr]) -> TypeExpr:
    if isinstance(tp, TypeOptional):
        return TypeOptional(inner=_resolve_type_dim_aliases(tp.inner, expr_defs))
    if isinstance(tp, TypeList):
        return TypeList(item=_resolve_type_dim_aliases(tp.item, expr_defs))
    if isinstance(tp, TypeTuple):
        return TypeTuple(
            items=tuple(_resolve_type_dim_aliases(item, expr_defs) for item in tp.items)
        )
    if isinstance(tp, TypeTensor):
        return TypeTensor(
            base=tp.base,
            dims=tuple(_resolve_dim_alias_token(dim, expr_defs) for dim in tp.dims),
        )
    if isinstance(tp, TypeNamed):
        return TypeNamed(
            name=tp.name,
            args=tuple(_resolve_dim_alias_token(dim, expr_defs) for dim in tp.args),
        )
    return tp


def _normalize_constraint_operand(
    operand: ConstraintAtom | tuple[ConstraintAtom, ...],
    ctx: _TcCtx,
    *,
    preferred_dim_roots: dict[str, str] | None = None,
) -> ConstraintAtom | tuple[ConstraintAtom, ...]:
    if isinstance(operand, tuple):
        normalized_items: list[ConstraintAtom] = []
        for item in operand:
            if isinstance(item, bool) or item is None:
                normalized_items.append(item)
            else:
                normalized_items.append(
                    _normalize_dim_token_for_module(item, ctx, preferred_dim_roots)
                )
        return tuple(normalized_items)
    if isinstance(operand, bool) or operand is None:
        return operand
    return _normalize_dim_token_for_module(operand, ctx, preferred_dim_roots)


def _normalize_constraint(
    constraint: Constraint,
    ctx: _TcCtx,
    *,
    preferred_dim_roots: dict[str, str] | None = None,
) -> Constraint:
    return Constraint(
        relation=constraint.relation,
        left=_normalize_constraint_operand(
            constraint.left, ctx, preferred_dim_roots=preferred_dim_roots
        ),
        right=(
            _normalize_constraint_operand(
                constraint.right, ctx, preferred_dim_roots=preferred_dim_roots
            )
            if constraint.right is not None
            else None
        ),
        guards=tuple(
            _normalize_constraint(item, ctx, preferred_dim_roots=preferred_dim_roots)
            for item in constraint.guards
        ),
    )


def _expr_to_constraint_operand(expr: AxonExpr) -> ConstraintAtom | None:
    if isinstance(expr, AxonExprAscribe):
        return _expr_to_constraint_operand(expr.expr)
    if isinstance(expr, AxonExprParen):
        return _expr_to_constraint_operand(expr.inner)
    if isinstance(expr, AxonExprInt):
        return expr.value
    if isinstance(expr, AxonExprBool):
        return expr.value
    if isinstance(expr, AxonExprNull):
        return None
    if isinstance(expr, AxonExprName):
        return expr.name
    if isinstance(expr, AxonExprBinary) and expr.op in {"+", "-", "*", "/"}:
        left = _expr_to_constraint_operand(expr.left)
        right = _expr_to_constraint_operand(expr.right)
        if left is None or right is None or isinstance(left, bool) or isinstance(right, bool):
            return None
        return DimExprBinary(op=expr.op, left=left, right=right)
    return None


def _condition_constraint_from_expr(cond: AxonExpr, *, truthy: bool) -> Constraint | None:
    if isinstance(cond, AxonExprName):
        return Constraint(relation="is_true" if truthy else "is_false", left=cond.name)
    if isinstance(cond, AxonExprBinary):
        if cond.op in {"==", "!="}:
            if isinstance(cond.left, AxonExprName) and isinstance(cond.right, AxonExprNull):
                return Constraint(
                    relation="is_null" if (truthy == (cond.op == "==")) else "not_null",
                    left=cond.left.name,
                )
            if isinstance(cond.right, AxonExprName) and isinstance(cond.left, AxonExprNull):
                return Constraint(
                    relation="is_null" if (truthy == (cond.op == "==")) else "not_null",
                    left=cond.right.name,
                )
            left = _expr_to_constraint_operand(cond.left)
            right = _expr_to_constraint_operand(cond.right)
            if left is not None and right is not None:
                relation = cond.op if truthy else ("!=" if cond.op == "==" else "==")
                return Constraint(relation=relation, left=left, right=right)
        if cond.op in _COMPARE_OPS:
            left = _expr_to_constraint_operand(cond.left)
            right = _expr_to_constraint_operand(cond.right)
            if left is None or right is None or isinstance(left, bool) or isinstance(right, bool):
                return None
            relation = cond.op
            if not truthy:
                relation = {
                    "==": "!=",
                    "!=": "==",
                    "<": ">=",
                    "<=": ">",
                    ">": "<=",
                    ">=": "<",
                }[relation]
            return Constraint(relation=relation, left=left, right=right)
    return None


def _collect_stmt_constraints(
    statements: tuple[AxonStatement, ...],
    *,
    out: list[Constraint],
    ctx: _TcCtx,
    guards: tuple[Constraint, ...] = (),
    condition_defs: dict[str, AxonExpr] | None = None,
) -> None:
    current_condition_defs = dict(condition_defs or {})

    def _append(constraint: Constraint) -> None:
        out.append(
            Constraint(
                relation=constraint.relation,
                left=constraint.left,
                right=constraint.right,
                guards=guards,
            )
        )

    for stmt in statements:
        if isinstance(stmt, AxonBind):
            if len(stmt.targets) == 1:
                if isinstance(stmt.expr, AxonExprBinary) and stmt.expr.op in {"==", "!="}:
                    current_condition_defs[stmt.targets[0]] = stmt.expr
                else:
                    current_condition_defs.pop(stmt.targets[0], None)
            if len(stmt.targets) == 1 and stmt.targets[0] != "_":
                if isinstance(stmt.expr, AxonExprTernary):
                    target = stmt.targets[0]
                    refine_expr = (
                        current_condition_defs.get(stmt.expr.cond.name, stmt.expr.cond)
                        if isinstance(stmt.expr.cond, AxonExprName)
                        else stmt.expr.cond
                    )
                    true_guard = _condition_constraint_from_expr(refine_expr, truthy=True)
                    false_guard = _condition_constraint_from_expr(refine_expr, truthy=False)
                    true_operand = _expr_to_constraint_operand(stmt.expr.true_expr)
                    false_operand = _expr_to_constraint_operand(stmt.expr.false_expr)
                    expr_type = _apply_subst(stmt.expr.inferred_type or TypeAny(), ctx)
                    if isinstance(
                        _expand_alias(expr_type, ctx), TypeDim | TypeInt | TypeBool | TypeNull
                    ):
                        if true_operand is not None and true_operand != target:
                            out.append(
                                Constraint(
                                    relation="=",
                                    left=target,
                                    right=true_operand,
                                    guards=guards if true_guard is None else (*guards, true_guard),
                                )
                            )
                        if false_operand is not None and false_operand != target:
                            out.append(
                                Constraint(
                                    relation="=",
                                    left=target,
                                    right=false_operand,
                                    guards=guards
                                    if false_guard is None
                                    else (*guards, false_guard),
                                )
                            )
                expr_type = _apply_subst(stmt.expr.inferred_type or TypeAny(), ctx)
                if isinstance(
                    _expand_alias(expr_type, ctx), TypeDim | TypeInt | TypeBool | TypeNull
                ):
                    operand = _expr_to_constraint_operand(stmt.expr)
                    if operand is not None and operand != stmt.targets[0]:
                        _append(Constraint(relation="=", left=stmt.targets[0], right=operand))
        elif isinstance(stmt, AxonCond):
            refine_expr = (
                current_condition_defs.get(stmt.cond.name, stmt.cond)
                if isinstance(stmt.cond, AxonExprName)
                else stmt.cond
            )
            true_guard = _condition_constraint_from_expr(refine_expr, truthy=True)
            false_guard = _condition_constraint_from_expr(refine_expr, truthy=False)
            _collect_stmt_constraints(
                stmt.true_body,
                out=out,
                ctx=ctx,
                guards=guards if true_guard is None else (*guards, true_guard),
                condition_defs=current_condition_defs,
            )
            _collect_stmt_constraints(
                stmt.false_body,
                out=out,
                ctx=ctx,
                guards=guards if false_guard is None else (*guards, false_guard),
                condition_defs=current_condition_defs,
            )
        elif isinstance(stmt, AxonRepeat):
            _collect_stmt_constraints(
                stmt.body, out=out, ctx=ctx, guards=guards, condition_defs=current_condition_defs
            )
        elif isinstance(stmt, AxonScopeBind):
            _collect_stmt_constraints(
                stmt.body, out=out, ctx=ctx, guards=guards, condition_defs=current_condition_defs
            )


def _collect_module_constraints(
    module: AxonDefinition,
    *,
    statements: tuple[AxonStatement, ...],
    ctx: _TcCtx,
) -> tuple[Constraint, ...]:
    constraints: list[Constraint] = list(module.constraints or ())
    for name, value in sorted(ctx.dim_substitutions.items()):
        if name.startswith(".."):
            continue
        normalized = _normalize_constraint_operand(value, ctx)
        if normalized != name:
            constraints.append(Constraint(relation="=", left=name, right=normalized))
    _collect_stmt_constraints(statements, out=constraints, ctx=ctx, condition_defs={})
    normalized_constraints = tuple(_normalize_constraint(item, ctx) for item in constraints)
    deduped: list[Constraint] = []
    seen: set[tuple[object, object, object, object]] = set()
    for item in normalized_constraints:
        key = (item.relation, item.left, item.right, item.guards)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return tuple(deduped)


def _constraint_operand_param_bindings(
    *,
    callee: AxonDefinition,
    args: tuple[AxonExpr, ...],
    kwargs: dict[str, AxonKwargValue],
) -> dict[ConstraintAtom, str]:
    bindings: dict[ConstraintAtom, str] = {}
    param_names = [*callee.path_params, *(param.name for param in callee.params)]
    for idx, arg in enumerate(args):
        if idx >= len(param_names) or idx < len(callee.path_params):
            continue
        operand = _expr_to_constraint_operand(arg)
        if isinstance(operand, str | DimExprBinary):
            bindings.setdefault(operand, param_names[idx])
    for key, raw_value in kwargs.items():
        if key not in param_names:
            continue
        param_idx = param_names.index(key)
        if param_idx < len(callee.path_params) or not isinstance(raw_value, AxonExpr):
            continue
        operand = _expr_to_constraint_operand(raw_value)
        if isinstance(operand, str | DimExprBinary):
            bindings.setdefault(operand, key)
    return bindings


def _call_param_constraints(
    *,
    callee: AxonDefinition,
    args: tuple[AxonExpr, ...],
    kwargs: dict[str, AxonKwargValue],
) -> tuple[Constraint, ...]:
    constraints: list[Constraint] = []
    param_names = [*callee.path_params, *(param.name for param in callee.params)]
    for idx, arg in enumerate(args):
        if idx >= len(param_names) or idx < len(callee.path_params):
            continue
        operand = _expr_to_constraint_operand(arg)
        if operand is None:
            continue
        constraints.append(Constraint(relation="=", left=param_names[idx], right=operand))
    for key, raw_value in kwargs.items():
        if key not in param_names or not isinstance(raw_value, AxonExpr):
            continue
        param_idx = param_names.index(key)
        if param_idx < len(callee.path_params):
            continue
        operand = _expr_to_constraint_operand(raw_value)
        if operand is None:
            continue
        constraints.append(Constraint(relation="=", left=key, right=operand))
    return tuple(constraints)


def _translate_constraint_operand(
    operand: ConstraintOperand,
    *,
    bindings: dict[ConstraintAtom, str],
) -> ConstraintOperand | None:
    if isinstance(operand, tuple):
        translated_items: list[ConstraintAtom] = []
        for item in operand:
            translated = _translate_constraint_operand(item, bindings=bindings)
            if translated is None or isinstance(translated, tuple):
                return None
            translated_items.append(translated)
        return tuple(translated_items)
    if isinstance(operand, bool) or operand is None or isinstance(operand, int):
        return operand
    if isinstance(operand, str):
        return bindings.get(operand, operand)
    translated_left = _translate_constraint_operand(operand.left, bindings=bindings)
    translated_right = _translate_constraint_operand(operand.right, bindings=bindings)
    if translated_left is None or translated_right is None:
        return None
    if isinstance(translated_left, tuple) or isinstance(translated_right, tuple):
        return None
    return DimExprBinary(op=operand.op, left=translated_left, right=translated_right)


def _translate_constraint_to_callee(
    constraint: Constraint,
    *,
    bindings: dict[ConstraintAtom, str],
) -> Constraint | None:
    left = _translate_constraint_operand(constraint.left, bindings=bindings)
    if left is None:
        return None
    right = (
        _translate_constraint_operand(constraint.right, bindings=bindings)
        if constraint.right is not None
        else None
    )
    if constraint.right is not None and right is None:
        return None
    translated_guards: list[Constraint] = []
    for guard in constraint.guards:
        translated_guard = _translate_constraint_to_callee(guard, bindings=bindings)
        if translated_guard is None:
            return None
        translated_guards.append(translated_guard)
    return Constraint(
        relation=constraint.relation,
        left=left,
        right=right,
        guards=tuple(translated_guards),
    )


def _collect_expr_call_constraints(
    expr: AxonExpr,
    *,
    caller: AxonDefinition,
    modules_by_name: dict[str, AxonDefinition],
    out: dict[str, list[Constraint]],
    guards: tuple[Constraint, ...],
    condition_defs: dict[str, AxonExpr],
    callsite_counter: list[int],
) -> None:
    if isinstance(expr, AxonExprCall):
        for arg in expr.args:
            _collect_expr_call_constraints(
                arg,
                caller=caller,
                modules_by_name=modules_by_name,
                out=out,
                guards=guards,
                condition_defs=condition_defs,
                callsite_counter=callsite_counter,
            )
        for raw_value in expr.kwargs.values():
            if isinstance(raw_value, AxonExpr):
                _collect_expr_call_constraints(
                    raw_value,
                    caller=caller,
                    modules_by_name=modules_by_name,
                    out=out,
                    guards=guards,
                    condition_defs=condition_defs,
                    callsite_counter=callsite_counter,
                )
        callee = modules_by_name.get(expr.callee)
        if callee is None:
            return
        bindings = _constraint_operand_param_bindings(
            callee=callee,
            args=expr.args,
            kwargs=expr.kwargs,
        )
        callsite_counter[0] += 1
        callsite_guard = Constraint(
            relation="callsite",
            left=f"{caller.name}->{callee.name}#{callsite_counter[0]}",
        )
        for param_constraint in _call_param_constraints(
            callee=callee,
            args=expr.args,
            kwargs=expr.kwargs,
        ):
            out.setdefault(callee.name, []).append(
                Constraint(
                    relation=param_constraint.relation,
                    left=param_constraint.left,
                    right=param_constraint.right,
                    guards=(callsite_guard,),
                )
            )
        if not guards or not bindings:
            return
        translated_guards: list[Constraint] = []
        for guard in guards:
            translated = _translate_constraint_to_callee(guard, bindings=bindings)
            if translated is not None:
                translated_guards.append(translated)
        if not translated_guards:
            return
        for translated in translated_guards:
            out.setdefault(callee.name, []).append(
                Constraint(
                    relation=translated.relation,
                    left=translated.left,
                    right=translated.right,
                    guards=(callsite_guard, *translated.guards),
                )
            )
        return
    if isinstance(expr, AxonExprBinary):
        _collect_expr_call_constraints(
            expr.left,
            caller=caller,
            modules_by_name=modules_by_name,
            out=out,
            guards=guards,
            condition_defs=condition_defs,
            callsite_counter=callsite_counter,
        )
        _collect_expr_call_constraints(
            expr.right,
            caller=caller,
            modules_by_name=modules_by_name,
            out=out,
            guards=guards,
            condition_defs=condition_defs,
            callsite_counter=callsite_counter,
        )
        return
    if isinstance(expr, AxonExprBind):
        _collect_expr_call_constraints(
            expr.value,
            caller=caller,
            modules_by_name=modules_by_name,
            out=out,
            guards=guards,
            condition_defs=condition_defs,
            callsite_counter=callsite_counter,
        )
        _collect_expr_call_constraints(
            expr.body,
            caller=caller,
            modules_by_name=modules_by_name,
            out=out,
            guards=guards,
            condition_defs=condition_defs,
            callsite_counter=callsite_counter,
        )
        return
    if isinstance(expr, AxonExprIf | AxonExprTernary):
        refine_expr = (
            condition_defs.get(expr.cond.name, expr.cond)
            if isinstance(expr.cond, AxonExprName)
            else expr.cond
        )
        true_guard = _condition_constraint_from_expr(refine_expr, truthy=True)
        false_guard = _condition_constraint_from_expr(refine_expr, truthy=False)
        _collect_expr_call_constraints(
            expr.cond,
            caller=caller,
            modules_by_name=modules_by_name,
            out=out,
            guards=guards,
            condition_defs=condition_defs,
            callsite_counter=callsite_counter,
        )
        _collect_expr_call_constraints(
            expr.true_expr,
            caller=caller,
            modules_by_name=modules_by_name,
            out=out,
            guards=guards if true_guard is None else (*guards, true_guard),
            condition_defs=condition_defs,
            callsite_counter=callsite_counter,
        )
        _collect_expr_call_constraints(
            expr.false_expr,
            caller=caller,
            modules_by_name=modules_by_name,
            out=out,
            guards=guards if false_guard is None else (*guards, false_guard),
            condition_defs=condition_defs,
            callsite_counter=callsite_counter,
        )
        return
    if isinstance(expr, AxonExprLambda):
        _collect_expr_call_constraints(
            expr.body,
            caller=caller,
            modules_by_name=modules_by_name,
            out=out,
            guards=guards,
            condition_defs=condition_defs,
            callsite_counter=callsite_counter,
        )
        return
    if isinstance(expr, AxonExprAscribe):
        _collect_expr_call_constraints(
            expr.expr,
            caller=caller,
            modules_by_name=modules_by_name,
            out=out,
            guards=guards,
            condition_defs=condition_defs,
            callsite_counter=callsite_counter,
        )
        return
    if isinstance(expr, AxonExprList | AxonExprTuple):
        for item in expr.items:
            _collect_expr_call_constraints(
                item,
                caller=caller,
                modules_by_name=modules_by_name,
                out=out,
                guards=guards,
                condition_defs=condition_defs,
                callsite_counter=callsite_counter,
            )
        return
    if isinstance(expr, AxonExprParen):
        _collect_expr_call_constraints(
            expr.inner,
            caller=caller,
            modules_by_name=modules_by_name,
            out=out,
            guards=guards,
            condition_defs=condition_defs,
            callsite_counter=callsite_counter,
        )
        return
    if isinstance(expr, AxonExprDo):
        _collect_stmt_call_constraints(
            expr.body,
            caller=caller,
            modules_by_name=modules_by_name,
            out=out,
            guards=guards,
            condition_defs=condition_defs,
            callsite_counter=callsite_counter,
        )


def _collect_stmt_call_constraints(
    statements: tuple[AxonStatement, ...],
    *,
    caller: AxonDefinition,
    modules_by_name: dict[str, AxonDefinition],
    out: dict[str, list[Constraint]],
    guards: tuple[Constraint, ...] = (),
    condition_defs: dict[str, AxonExpr] | None = None,
    callsite_counter: list[int] | None = None,
) -> None:
    current_condition_defs = dict(condition_defs or {})
    current_counter = callsite_counter if callsite_counter is not None else [0]
    for stmt in statements:
        if isinstance(stmt, AxonBind):
            if len(stmt.targets) == 1:
                if isinstance(stmt.expr, AxonExprBinary) and stmt.expr.op in {"==", "!="}:
                    current_condition_defs[stmt.targets[0]] = stmt.expr
                else:
                    current_condition_defs.pop(stmt.targets[0], None)
            _collect_expr_call_constraints(
                stmt.expr,
                caller=caller,
                modules_by_name=modules_by_name,
                out=out,
                guards=guards,
                condition_defs=current_condition_defs,
                callsite_counter=current_counter,
            )
        elif isinstance(stmt, AxonReturn | AxonYield):
            for value in stmt.values:
                _collect_expr_call_constraints(
                    value,
                    caller=caller,
                    modules_by_name=modules_by_name,
                    out=out,
                    guards=guards,
                    condition_defs=current_condition_defs,
                    callsite_counter=current_counter,
                )
        elif isinstance(stmt, AxonCond):
            refine_expr = (
                current_condition_defs.get(stmt.cond.name, stmt.cond)
                if isinstance(stmt.cond, AxonExprName)
                else stmt.cond
            )
            true_guard = _condition_constraint_from_expr(refine_expr, truthy=True)
            false_guard = _condition_constraint_from_expr(refine_expr, truthy=False)
            _collect_expr_call_constraints(
                stmt.cond,
                caller=caller,
                modules_by_name=modules_by_name,
                out=out,
                guards=guards,
                condition_defs=current_condition_defs,
                callsite_counter=current_counter,
            )
            _collect_stmt_call_constraints(
                stmt.true_body,
                caller=caller,
                modules_by_name=modules_by_name,
                out=out,
                guards=guards if true_guard is None else (*guards, true_guard),
                condition_defs=current_condition_defs,
                callsite_counter=current_counter,
            )
            _collect_stmt_call_constraints(
                stmt.false_body,
                caller=caller,
                modules_by_name=modules_by_name,
                out=out,
                guards=guards if false_guard is None else (*guards, false_guard),
                condition_defs=current_condition_defs,
                callsite_counter=current_counter,
            )
        elif isinstance(stmt, AxonRepeat):
            _collect_expr_call_constraints(
                stmt.from_expr,
                caller=caller,
                modules_by_name=modules_by_name,
                out=out,
                guards=guards,
                condition_defs=current_condition_defs,
                callsite_counter=current_counter,
            )
            _collect_expr_call_constraints(
                stmt.to_expr,
                caller=caller,
                modules_by_name=modules_by_name,
                out=out,
                guards=guards,
                condition_defs=current_condition_defs,
                callsite_counter=current_counter,
            )
            _collect_expr_call_constraints(
                stmt.step_expr,
                caller=caller,
                modules_by_name=modules_by_name,
                out=out,
                guards=guards,
                condition_defs=current_condition_defs,
                callsite_counter=current_counter,
            )
            _collect_stmt_call_constraints(
                stmt.body,
                caller=caller,
                modules_by_name=modules_by_name,
                out=out,
                guards=guards,
                condition_defs=current_condition_defs,
                callsite_counter=current_counter,
            )
        elif isinstance(stmt, AxonScopeBind):
            for raw_value in stmt.kwargs.values():
                if isinstance(raw_value, AxonExpr):
                    _collect_expr_call_constraints(
                        raw_value,
                        caller=caller,
                        modules_by_name=modules_by_name,
                        out=out,
                        guards=guards,
                        condition_defs=current_condition_defs,
                        callsite_counter=current_counter,
                    )
            _collect_stmt_call_constraints(
                stmt.body,
                caller=caller,
                modules_by_name=modules_by_name,
                out=out,
                guards=guards,
                condition_defs=current_condition_defs,
                callsite_counter=current_counter,
            )


def _thread_interprocedural_call_guards(program: AxonFile) -> AxonFile:
    modules_by_name = {module.name: module for module in program.modules}
    imported_constraints: dict[str, list[Constraint]] = {}
    for module in program.modules:
        _collect_stmt_call_constraints(
            module.statements,
            caller=module,
            modules_by_name=modules_by_name,
            out=imported_constraints,
        )
    rewritten_modules: list[AxonDefinition] = []
    for module in program.modules:
        merged_constraints = list(module.constraints or ())
        merged_constraints.extend(imported_constraints.get(module.name, ()))
        deduped: list[Constraint] = []
        seen: set[tuple[object, object, object, object]] = set()
        for item in merged_constraints:
            key = (item.relation, item.left, item.right, item.guards)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        rewritten_modules.append(replace(module, constraints=tuple(deduped)))
    return replace(program, modules=tuple(rewritten_modules))


def _bind_dim_tokens(
    expected: tuple[DimToken, ...],
    actual: tuple[DimToken, ...],
    *,
    subst: dict[str, DimToken | tuple[DimToken, ...]],
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


def _collect_module_call_dim_subst(
    *,
    expected_type: TypeExpr | None,
    actual_type: TypeExpr,
    actual_expr: AxonExpr,
    ctx: _TcCtx,
    subst: dict[str, DimToken | tuple[DimToken, ...]],
) -> None:
    if expected_type is None:
        return
    expected_expanded = _expand_alias(expected_type, ctx)
    actual_expanded = _expand_alias(actual_type, ctx)
    if isinstance(expected_expanded, TypeOptional):
        _collect_module_call_dim_subst(
            expected_type=expected_expanded.inner,
            actual_type=actual_type,
            actual_expr=actual_expr,
            ctx=ctx,
            subst=subst,
        )
        return
    if isinstance(expected_expanded, TypeDim):
        dim_token = _expr_to_dim_token(actual_expr)
        if dim_token is not None:
            if isinstance(actual_expr, AxonExprName):
                subst.setdefault(actual_expr.name, dim_token)
        return
    if isinstance(expected_expanded, TypeTensor) and isinstance(actual_expanded, TypeTensor):
        _bind_dim_tokens(expected_expanded.dims, actual_expanded.dims, subst=subst)
        return


def _collect_instantiated_dim_names(
    raw_type: TypeExpr | None,
    instantiated_type: TypeExpr | None,
    *,
    out: dict[str, DimToken],
) -> None:
    if raw_type is None or instantiated_type is None:
        return
    if isinstance(raw_type, TypeOptional) and isinstance(instantiated_type, TypeOptional):
        _collect_instantiated_dim_names(raw_type.inner, instantiated_type.inner, out=out)
        return
    if isinstance(raw_type, TypeList) and isinstance(instantiated_type, TypeList):
        _collect_instantiated_dim_names(raw_type.item, instantiated_type.item, out=out)
        return
    if isinstance(raw_type, TypeTuple) and isinstance(instantiated_type, TypeTuple):
        for raw_item, instantiated_item in zip(
            raw_type.items, instantiated_type.items, strict=False
        ):
            _collect_instantiated_dim_names(raw_item, instantiated_item, out=out)
        return
    if isinstance(raw_type, TypeTensor) and isinstance(instantiated_type, TypeTensor):
        if len(raw_type.dims) != len(instantiated_type.dims):
            return
        for raw_dim, instantiated_dim in zip(
            raw_type.dims, instantiated_type.dims, strict=True
        ):
            if isinstance(raw_dim, str) and isinstance(instantiated_dim, str):
                out.setdefault(raw_dim, instantiated_dim)
            elif isinstance(raw_dim, DimExprBinary) and isinstance(
                instantiated_dim, DimExprBinary
            ):
                _collect_instantiated_dim_names(
                    TypeTensor(base="Tensor", dims=(raw_dim.left, raw_dim.right)),
                    TypeTensor(
                        base="Tensor",
                        dims=(instantiated_dim.left, instantiated_dim.right),
                    ),
                    out=out,
                )
        return
    if isinstance(raw_type, TypeNamed) and isinstance(instantiated_type, TypeNamed):
        if raw_type.name != instantiated_type.name or len(raw_type.args) != len(
            instantiated_type.args
        ):
            return
        for raw_dim, instantiated_dim in zip(
            raw_type.args, instantiated_type.args, strict=True
        ):
            if isinstance(raw_dim, str) and isinstance(instantiated_dim, str):
                out.setdefault(raw_dim, instantiated_dim)


def _module_dim_instantiation_map(
    module: AxonDefinition,
    return_types: list[TypeExpr],
) -> dict[str, DimToken]:
    out: dict[str, DimToken] = {}
    if module.return_type_expr is None:
        return out
    instantiated_return: TypeExpr = (
        TypeTuple(items=tuple(return_types)) if len(return_types) != 1 else return_types[0]
    )
    _collect_instantiated_dim_names(module.return_type_expr, instantiated_return, out=out)
    return out


def _primitive_op_name(callee: str) -> str | None:
    if not callee.startswith("_"):
        return None
    return callee[1:]


def _type_expr_from_spec(spec: str, *, ctx: _TcCtx, module_name: str) -> TypeExpr:
    return _scoped_typevars(
        parse_type_expr(spec), module_name=module_name, ctx=ctx, freshen_generics=True
    )


def _collect_type_dim_names(tp: TypeExpr | None) -> set[str]:
    if tp is None:
        return set()
    if isinstance(tp, TypeOptional):
        return _collect_type_dim_names(tp.inner)
    if isinstance(tp, TypeList):
        return _collect_type_dim_names(tp.item)
    if isinstance(tp, TypeTuple):
        tuple_names: set[str] = set()
        for item in tp.items:
            tuple_names.update(_collect_type_dim_names(item))
        return tuple_names
    if isinstance(tp, TypeTensor):
        tensor_names: set[str] = set()
        for dim in tp.dims:
            tensor_names.update(name for name in dim_token_names(dim) if name.isidentifier())
        return tensor_names
    if isinstance(tp, TypeNamed):
        return {name for name in tp.args if isinstance(name, str) and name.isidentifier()}
    return set()


def _preferred_module_dim_names(
    raw_params: tuple[AxonParam, ...], refined_return: TypeExpr | None
) -> set[str]:
    names = _collect_type_dim_names(refined_return)
    for param in raw_params:
        names.update(_collect_type_dim_names(param.type_expr))
        if isinstance(param.type_expr, TypeDim) or (
            isinstance(param.type_expr, TypeOptional) and isinstance(param.type_expr.inner, TypeDim)
        ):
            names.add(param.name)
    return names


def _collect_statement_dim_names(statements: tuple[AxonStatement, ...]) -> set[str]:
    names: set[str] = set()

    def _visit_expr(expr: AxonExpr) -> None:
        if isinstance(expr, AxonExprName) and isinstance(expr.inferred_type, TypeDim):
            names.add(expr.name)
            return
        if isinstance(expr, AxonExprCall):
            for arg in expr.args:
                _visit_expr(arg)
            for value in expr.kwargs.values():
                if isinstance(value, AxonExpr):
                    _visit_expr(value)
            return
        if isinstance(expr, AxonExprList | AxonExprTuple):
            for item in expr.items:
                _visit_expr(item)
            return
        if isinstance(expr, AxonExprAscribe):
            _visit_expr(expr.expr)
            return
        if isinstance(expr, AxonExprBinary):
            _visit_expr(expr.left)
            _visit_expr(expr.right)
            return
        if isinstance(expr, AxonExprIf | AxonExprTernary):
            _visit_expr(expr.cond)
            _visit_expr(expr.true_expr)
            _visit_expr(expr.false_expr)
            return
        if isinstance(expr, AxonExprBind):
            _visit_expr(expr.value)
            _visit_expr(expr.body)
            return
        if isinstance(expr, AxonExprDo):
            for stmt in expr.body:
                _visit_stmt(stmt)
            return
        if isinstance(expr, AxonExprLambda):
            _visit_expr(expr.body)
            return
        if isinstance(expr, AxonExprParen):
            _visit_expr(expr.inner)
            return
        if isinstance(expr, AxonExprPipe):
            _visit_expr(expr.value)
            for stage in expr.stages:
                _visit_expr(stage)

    def _visit_stmt(stmt: AxonStatement) -> None:
        if isinstance(stmt, AxonBind):
            _visit_expr(stmt.expr)
            return
        if isinstance(stmt, AxonReturn | AxonYield):
            for value in stmt.values:
                _visit_expr(value)
            return
        if isinstance(stmt, AxonCond):
            _visit_expr(stmt.cond)
            for item in stmt.true_body:
                _visit_stmt(item)
            for item in stmt.false_body:
                _visit_stmt(item)
            return
        if isinstance(stmt, AxonRepeat):
            _visit_expr(stmt.from_expr)
            _visit_expr(stmt.to_expr)
            _visit_expr(stmt.step_expr)
            for item in stmt.body:
                _visit_stmt(item)
            return
        if isinstance(stmt, AxonScopeBind):
            for raw_value in stmt.kwargs.values():
                if isinstance(raw_value, AxonExpr):
                    _visit_expr(raw_value)
            for item in stmt.body:
                _visit_stmt(item)

    for statement in statements:
        _visit_stmt(statement)
    return names


def _bind_header_dim_preferences_from_tokens(
    expected: DimToken,
    actual: DimToken,
    *,
    subst: dict[str, DimToken],
) -> None:
    if isinstance(expected, str):
        if _is_generated_dim_name(expected):
            subst.setdefault(expected, actual)
        return
    if isinstance(expected, int):
        return
    if isinstance(actual, DimExprBinary) and expected.op == actual.op:
        _bind_header_dim_preferences_from_tokens(expected.left, actual.left, subst=subst)
        _bind_header_dim_preferences_from_tokens(expected.right, actual.right, subst=subst)


def _bind_header_dim_preferences_from_types(
    expected: TypeExpr | None,
    actual: TypeExpr | None,
    *,
    subst: dict[str, DimToken],
) -> None:
    if expected is None or actual is None:
        return
    if isinstance(expected, TypeOptional):
        inner_actual = actual.inner if isinstance(actual, TypeOptional) else actual
        _bind_header_dim_preferences_from_types(expected.inner, inner_actual, subst=subst)
        return
    if isinstance(actual, TypeOptional):
        _bind_header_dim_preferences_from_types(expected, actual.inner, subst=subst)
        return
    if isinstance(expected, TypeTensor) and isinstance(actual, TypeTensor):
        if len(expected.dims) != len(actual.dims):
            return
        for left, right in zip(expected.dims, actual.dims, strict=True):
            _bind_header_dim_preferences_from_tokens(left, right, subst=subst)
        return
    if isinstance(expected, TypeNamed) and isinstance(actual, TypeNamed):
        if len(expected.args) != len(actual.args):
            return
        for left, right in zip(expected.args, actual.args, strict=True):
            _bind_header_dim_preferences_from_tokens(left, right, subst=subst)
        return
    if isinstance(expected, TypeList) and isinstance(actual, TypeList):
        _bind_header_dim_preferences_from_types(expected.item, actual.item, subst=subst)
        return
    if (
        isinstance(expected, TypeTuple)
        and isinstance(actual, TypeTuple)
        and len(expected.items) == len(actual.items)
    ):
        for expected_item, actual_item in zip(expected.items, actual.items, strict=True):
            _bind_header_dim_preferences_from_types(expected_item, actual_item, subst=subst)


def _collect_header_dim_preferences_from_param_uses(
    raw_params: tuple[AxonParam, ...],
    statements: tuple[AxonStatement, ...],
) -> dict[str, DimToken]:
    params_by_name = {param.name: param for param in raw_params}
    subst: dict[str, DimToken] = {}

    def _visit_expr(expr: AxonExpr) -> None:
        if isinstance(expr, AxonExprName):
            param = params_by_name.get(expr.name)
            if param is not None:
                _bind_header_dim_preferences_from_types(
                    param.type_expr, expr.inferred_type, subst=subst
                )
            return
        if isinstance(expr, AxonExprCall):
            for arg in expr.args:
                _visit_expr(arg)
            for value in expr.kwargs.values():
                if isinstance(value, AxonExpr):
                    _visit_expr(value)
            return
        if isinstance(expr, AxonExprList | AxonExprTuple):
            for item in expr.items:
                _visit_expr(item)
            return
        if isinstance(expr, AxonExprAscribe):
            _visit_expr(expr.expr)
            return
        if isinstance(expr, AxonExprBinary):
            _visit_expr(expr.left)
            _visit_expr(expr.right)
            return
        if isinstance(expr, AxonExprIf | AxonExprTernary):
            _visit_expr(expr.cond)
            _visit_expr(expr.true_expr)
            _visit_expr(expr.false_expr)
            return
        if isinstance(expr, AxonExprBind):
            _visit_expr(expr.value)
            _visit_expr(expr.body)
            return
        if isinstance(expr, AxonExprDo):
            for stmt in expr.body:
                _visit_stmt(stmt)
            return
        if isinstance(expr, AxonExprLambda):
            _visit_expr(expr.body)
            return
        if isinstance(expr, AxonExprParen):
            _visit_expr(expr.inner)
            return
        if isinstance(expr, AxonExprPipe):
            _visit_expr(expr.value)
            for stage in expr.stages:
                _visit_expr(stage)

    def _visit_stmt(stmt: AxonStatement) -> None:
        if isinstance(stmt, AxonBind):
            _visit_expr(stmt.expr)
            return
        if isinstance(stmt, AxonReturn | AxonYield):
            for value in stmt.values:
                _visit_expr(value)
            return
        if isinstance(stmt, AxonCond):
            _visit_expr(stmt.cond)
            for item in stmt.true_body:
                _visit_stmt(item)
            for item in stmt.false_body:
                _visit_stmt(item)
            return
        if isinstance(stmt, AxonRepeat):
            _visit_expr(stmt.from_expr)
            _visit_expr(stmt.to_expr)
            _visit_expr(stmt.step_expr)
            for item in stmt.body:
                _visit_stmt(item)
            return
        if isinstance(stmt, AxonScopeBind):
            for raw_value in stmt.kwargs.values():
                if isinstance(raw_value, AxonExpr):
                    _visit_expr(raw_value)
            for item in stmt.body:
                _visit_stmt(item)

    for statement in statements:
        _visit_stmt(statement)
    return subst


def _collect_param_type_refinements_from_uses(
    raw_params: tuple[AxonParam, ...],
    statements: tuple[AxonStatement, ...],
    ctx: _TcCtx,
) -> dict[str, TypeExpr]:
    params_by_name = {param.name: param for param in raw_params}
    active = set(params_by_name)
    refinements: dict[str, TypeExpr] = {}

    def _record(name: str, tp: TypeExpr | None) -> None:
        if tp is None or name not in active:
            return
        param = params_by_name[name]
        refined = _apply_subst(tp, ctx)
        if param.optional and isinstance(refined, TypeOptional):
            refined = refined.inner
        current = refinements.get(name)
        refinements[name] = refined if current is None else _join_branch_types(current, refined, ctx)

    def _visit_expr(expr: AxonExpr) -> None:
        if isinstance(expr, AxonExprName):
            _record(expr.name, expr.inferred_type)
            return
        if isinstance(expr, AxonExprCall):
            for arg in expr.args:
                _visit_expr(arg)
            for value in expr.kwargs.values():
                if isinstance(value, AxonExpr):
                    _visit_expr(value)
            return
        if isinstance(expr, AxonExprList | AxonExprTuple):
            for item in expr.items:
                _visit_expr(item)
            return
        if isinstance(expr, AxonExprAscribe):
            _visit_expr(expr.expr)
            return
        if isinstance(expr, AxonExprBinary):
            _visit_expr(expr.left)
            _visit_expr(expr.right)
            return
        if isinstance(expr, AxonExprIf | AxonExprTernary):
            _visit_expr(expr.cond)
            _visit_expr(expr.true_expr)
            _visit_expr(expr.false_expr)
            return
        if isinstance(expr, AxonExprBind):
            _visit_expr(expr.value)
            _visit_expr(expr.body)
            return
        if isinstance(expr, AxonExprDo):
            for stmt in expr.body:
                _visit_stmt(stmt)
            return
        if isinstance(expr, AxonExprLambda):
            _visit_expr(expr.body)
            return
        if isinstance(expr, AxonExprParen):
            _visit_expr(expr.inner)
            return
        if isinstance(expr, AxonExprPipe):
            _visit_expr(expr.value)
            for stage in expr.stages:
                _visit_expr(stage)

    def _visit_stmt(stmt: AxonStatement) -> None:
        if isinstance(stmt, AxonBind):
            _visit_expr(stmt.expr)
            for target in stmt.targets:
                active.discard(target)
            return
        if isinstance(stmt, AxonReturn | AxonYield):
            for value in stmt.values:
                _visit_expr(value)
            return
        if isinstance(stmt, AxonCond):
            _visit_expr(stmt.cond)
            for item in stmt.true_body:
                _visit_stmt(item)
            for item in stmt.false_body:
                _visit_stmt(item)
            return
        if isinstance(stmt, AxonRepeat):
            _visit_expr(stmt.from_expr)
            _visit_expr(stmt.to_expr)
            _visit_expr(stmt.step_expr)
            for item in stmt.body:
                _visit_stmt(item)
            return
        if isinstance(stmt, AxonScopeBind):
            for raw_value in stmt.kwargs.values():
                if isinstance(raw_value, AxonExpr):
                    _visit_expr(raw_value)
            for item in stmt.body:
                _visit_stmt(item)

    for statement in statements:
        _visit_stmt(statement)
    return refinements


@dataclass(frozen=True)
class _PrimitiveTypeHelpers:
    type_dims: Any
    expr_to_dim_token: Any
    type_tensor: Any
    resolve_name_expr: Any
    broadcast_tensor_dims: Any
    dim_equivalent: Any


def _is_type_expr_instance(value: object) -> bool:
    return isinstance(
        value,
        (
            TypeAny,
            TypeInt,
            TypeFloat,
            TypeBool,
            TypeNull,
            TypeString,
            TypePath,
            TypeDim,
            TypeVar,
            TypeNamed,
            TypeOptional,
            TypeTensor,
            TypeList,
            TypeTuple,
        ),
    )


def _infer_primitive_call(
    *,
    callee: str,
    typed_args: list[AxonExpr],
    arg_types: list[TypeExpr],
    typed_kwargs: dict[str, AxonKwargValue],
    kwarg_types: dict[str, TypeExpr],
    ctx: _TcCtx,
    module_name: str,
    expected_arity: int | None,
    expr_defs: dict[str, AxonExpr],
) -> tuple[AxonExpr, TypeExpr, int] | None:
    op_name = _primitive_op_name(callee)
    if op_name is None:
        return None
    type_signature = get_op_lowering_type_signature(op_name)
    if type_signature is None:
        return None
    typed_call = replace(
        AxonExprCall(callee=callee, args=tuple(typed_args), kwargs=typed_kwargs),
        inferred_type=None,
        inferred_arity=None,
        inferred_dims=None,
    )
    arg_specs = type_signature.get("args")
    if isinstance(arg_specs, tuple):
        for arg_tp, spec in zip(arg_types, arg_specs, strict=False):
            if isinstance(spec, str):
                _unify(
                    arg_tp, _type_expr_from_spec(spec, ctx=ctx, module_name=f"_op::{op_name}"), ctx
                )
    kwarg_specs = type_signature.get("kwargs")
    if isinstance(kwarg_specs, dict):
        for key, value_tp in kwarg_types.items():
            spec = kwarg_specs.get(key)
            if isinstance(spec, str):
                _unify(
                    value_tp,
                    _type_expr_from_spec(spec, ctx=ctx, module_name=f"_op::{op_name}"),
                    ctx,
                )
    type_rule = get_op_type_rule(op_name)
    if type_rule is not None:
        inferred = type_rule(
            arg_types=tuple(arg_types),
            kwarg_types=dict(kwarg_types),
            args=tuple(typed_args),
            kwargs=dict(typed_kwargs),
            helpers=_PrimitiveTypeHelpers(
                type_dims=lambda tp: _type_dims(tp, ctx),
                expr_to_dim_token=lambda expr: _expr_to_dim_token_resolved(expr, expr_defs)
                if isinstance(expr, AxonExpr)
                else None,
                type_tensor=lambda *, dims: TypeTensor(base="Tensor", dims=tuple(dims)),
                resolve_name_expr=lambda name: expr_defs.get(name),
                broadcast_tensor_dims=lambda left, right: _unify_broadcast_tensor_dims(
                    tuple(left), tuple(right), ctx
                ),
                dim_equivalent=lambda left, right: _dim_equivalent(left, right, ctx),
            ),
        )
        if _is_type_expr_instance(inferred):
            inferred = _resolve_type_dim_aliases(inferred, expr_defs)
            arity = len(inferred.items) if isinstance(inferred, TypeTuple) else 1
            return _annotate_expr(typed_call, inferred, arity=arity, ctx=ctx), inferred, arity
    returns_spec = type_signature.get("returns")
    if (
        isinstance(returns_spec, tuple)
        and len(returns_spec) == 1
        and isinstance(returns_spec[0], str)
    ):
        result_tp = _type_expr_from_spec(returns_spec[0], ctx=ctx, module_name=f"_op::{op_name}")
        result_tp = _resolve_type_dim_aliases(result_tp, expr_defs)
        return _annotate_expr(typed_call, result_tp, arity=1, ctx=ctx), result_tp, 1
    if isinstance(returns_spec, tuple) and all(isinstance(item, str) for item in returns_spec):
        result_tp = TypeTuple(
            items=tuple(
                _resolve_type_dim_aliases(
                    _type_expr_from_spec(item, ctx=ctx, module_name=f"_op::{op_name}"),
                    expr_defs,
                )
                for item in returns_spec
            )
        )
        return _annotate_expr(typed_call, result_tp, arity=len(result_tp.items), ctx=ctx), result_tp, len(result_tp.items)
    return None


def _implicit_leading_path_param_count(
    callee: AxonDefinition, arg_types: tuple[TypeExpr, ...] | list[TypeExpr], ctx: _TcCtx
) -> int:
    leading_path_count = len(callee.path_params)
    for param in callee.params:
        if param.type_expr is None:
            break
        if isinstance(_expand_alias(_apply_subst(param.type_expr, ctx), ctx), TypePath):
            leading_path_count += 1
            continue
        break
    if leading_path_count == 0:
        return 0
    if not arg_types:
        return leading_path_count
    first_arg = _expand_alias(_apply_subst(arg_types[0], ctx), ctx)
    if isinstance(first_arg, TypePath):
        return 0
    return leading_path_count


def _refined_definition_call_result_type(
    *,
    callee: AxonDefinition,
    typed_args: tuple[AxonExpr, ...],
    arg_types: tuple[TypeExpr, ...],
    typed_kwargs: dict[str, AxonKwargValue],
    kwarg_types: dict[str, TypeExpr],
    ctx: _TcCtx,
    module_name: str,
    recursive_env: _RecursiveInterfaces | None,
    expected_arity: int | None,
) -> TypeExpr | None:
    if callee.name == module_name:
        return None
    if recursive_env is not None and module_name in recursive_env.members:
        return None
    if recursive_env is not None and callee.name in recursive_env.members:
        return None
    param_names = [*callee.path_params, *(param.name for param in callee.params)]
    implicit_path_count = _implicit_leading_path_param_count(callee, arg_types, ctx)
    positional_param_names = param_names[implicit_path_count:]
    env: dict[str, TypeExpr] = {}
    expr_defs: dict[str, AxonExpr] = {}
    omitted_null_dim_params: set[str] = set()
    for path_param in callee.path_params:
        env[path_param] = TypePath()
    for param in callee.params:
        env[param.name] = param.type_expr or TypeAny()
        if param.optional:
            env[param.name] = TypeOptional(inner=env[param.name])
        is_optional_null_dim = (
            param.optional
            and (
                isinstance(param.type_expr, TypeDim)
                or (
                    isinstance(param.type_expr, TypeOptional)
                    and isinstance(param.type_expr.inner, TypeDim)
                )
            )
            and isinstance(param.default_expr, AxonExprNull)
        )
        if is_optional_null_dim:
            omitted_null_dim_params.add(param.name)
        elif param.default_expr is not None:
            expr_defs[param.name] = param.default_expr
    for idx, (arg, arg_type) in enumerate(zip(typed_args, arg_types, strict=False)):
        if idx >= len(positional_param_names):
            continue
        name = positional_param_names[idx]
        env[name] = _apply_subst(arg_type, ctx)
        expr_defs[name] = arg
    for key, value in typed_kwargs.items():
        if not isinstance(value, AxonExpr) or key not in kwarg_types:
            continue
        env[key] = _apply_subst(kwarg_types[key], ctx)
        expr_defs[key] = value
        omitted_null_dim_params.discard(key)
    local_ctx = ctx.child(share_constraints=False)
    try:
        typed_statements, _, body_returns = _infer_statements(
            callee.statements,
            env=env,
            expr_defs=expr_defs,
            condition_defs={},
            ctx=local_ctx,
            module_name=callee.name,
            recursive_env=recursive_env,
            expected_return_types=None,
            in_loop=False,
        )
        del typed_statements
        if body_returns:
            result = _module_return_from_body_returns(body_returns, local_ctx)
            if result is not None and omitted_null_dim_params & _collect_type_dim_names(result):
                return None
            return result
        if callee.body_expr is not None:
            _, body_type, body_arity = _infer_expr(
                callee.body_expr,
                env=env,
                expr_defs=expr_defs,
                ctx=local_ctx,
                module_name=callee.name,
                recursive_env=recursive_env,
                expected_arity=expected_arity,
            )
            if body_arity > 1 and not isinstance(body_type, TypeTuple):
                return TypeTuple(items=(body_type,))
            body_type = _apply_subst(body_type, local_ctx)
            if omitted_null_dim_params & _collect_type_dim_names(body_type):
                return None
            return body_type
    except ValueError:
        return None
    return None


def _annotate_expr(expr: AxonExpr, tp: TypeExpr, *, arity: int, ctx: _TcCtx) -> AxonExpr:
    inferred_type = _apply_subst(tp, ctx)
    inferred_arity = arity
    if isinstance(inferred_type, TypeTuple):
        inferred_arity = len(inferred_type.items)
    return replace(
        expr,
        inferred_type=inferred_type,
        inferred_arity=inferred_arity,
        inferred_dims=_type_dims(tp, ctx),
    )


def _retag_numeric_literals(expr: AxonExpr, expected: TypeExpr, ctx: _TcCtx) -> AxonExpr:
    expected_tp = _expand_alias(_apply_subst(expected, ctx), ctx)
    if isinstance(expected_tp, TypeOptional):
        expected_tp = expected_tp.inner
    if isinstance(expr, AxonExprInt):
        if isinstance(expected_tp, TypeDim):
            return _annotate_expr(expr, TypeDim(), arity=1, ctx=ctx)
        if isinstance(expected_tp, TypeFloat):
            return _annotate_expr(expr, TypeFloat(), arity=1, ctx=ctx)
        return expr
    if isinstance(expr, AxonExprFloat):
        if isinstance(expected_tp, TypeFloat):
            return _annotate_expr(expr, TypeFloat(), arity=1, ctx=ctx)
        return expr
    if isinstance(expr, AxonExprParen):
        inner = _retag_numeric_literals(expr.inner, expected_tp, ctx)
        return _annotate_expr(replace(expr, inner=inner), expected_tp, arity=1, ctx=ctx)
    if isinstance(expr, AxonExprAscribe):
        inner = _retag_numeric_literals(expr.expr, expected_tp, ctx)
        return _annotate_expr(replace(expr, expr=inner), expected_tp, arity=1, ctx=ctx)
    if isinstance(expr, AxonExprList) and isinstance(expected_tp, TypeList):
        items = tuple(_retag_numeric_literals(item, expected_tp.item, ctx) for item in expr.items)
        return _annotate_expr(replace(expr, items=items), expected_tp, arity=1, ctx=ctx)
    if (
        isinstance(expr, AxonExprTuple)
        and isinstance(expected_tp, TypeTuple)
        and len(expr.items) == len(expected_tp.items)
    ):
        items = tuple(
            _retag_numeric_literals(item, item_tp, ctx)
            for item, item_tp in zip(expr.items, expected_tp.items, strict=True)
        )
        return _annotate_expr(replace(expr, items=items), expected_tp, arity=len(items), ctx=ctx)
    if isinstance(expr, AxonExprBinary) and expr.op in _ARITH_OPS:
        left = _retag_numeric_literals(expr.left, expected_tp, ctx)
        right = _retag_numeric_literals(expr.right, expected_tp, ctx)
        return _annotate_expr(replace(expr, left=left, right=right), expected_tp, arity=1, ctx=ctx)
    return expr


def _destructure_type(tp: TypeExpr, arity: int, ctx: _TcCtx) -> tuple[TypeExpr, ...]:
    tp = _apply_subst(tp, ctx)
    expanded = _expand_alias(tp, ctx)
    if arity == 1:
        return (expanded,)
    if isinstance(expanded, TypeTuple):
        if len(expanded.items) != arity:
            raise ValueError("Axon typecheck failed: tuple bind arity mismatch")
        return expanded.items
    if isinstance(expanded, TypeList):
        return tuple(expanded.item for _ in range(arity))
    if isinstance(expanded, TypeVar):
        items = tuple(ctx.fresh_type_var() for _ in range(arity))
        _unify(expanded, TypeTuple(items=items), ctx)
        return items
    raise ValueError("Axon typecheck failed: cannot destructure non-tuple/list expression")


def _infer_expr(
    expr: AxonExpr,
    *,
    env: dict[str, TypeExpr],
    expr_defs: dict[str, AxonExpr],
    ctx: _TcCtx,
    module_name: str,
    recursive_env: _RecursiveInterfaces | None = None,
    expected_arity: int | None = None,
) -> tuple[AxonExpr, TypeExpr, int]:
    if isinstance(expr, AxonExprName):
        tp = env.get(expr.name)
        if tp is None:
            raise ValueError(
                f"Axon typecheck failed in module {module_name!r}: "
                f"untyped unresolved name {expr.name!r}"
            )
        return _annotate_expr(expr, tp, arity=1, ctx=ctx), tp, 1
    if isinstance(expr, AxonExprInt):
        return _annotate_expr(expr, TypeInt(), arity=1, ctx=ctx), TypeInt(), 1
    if isinstance(expr, AxonExprFloat):
        return _annotate_expr(expr, TypeFloat(), arity=1, ctx=ctx), TypeFloat(), 1
    if isinstance(expr, AxonExprBool):
        return _annotate_expr(expr, TypeBool(), arity=1, ctx=ctx), TypeBool(), 1
    if isinstance(expr, AxonExprNull):
        return _annotate_expr(expr, TypeNull(), arity=1, ctx=ctx), TypeNull(), 1
    if isinstance(expr, AxonExprString):
        return _annotate_expr(expr, TypeString(), arity=1, ctx=ctx), TypeString(), 1
    if isinstance(expr, AxonExprPath):
        return _annotate_expr(expr, TypePath(), arity=1, ctx=ctx), TypePath(), 1
    if isinstance(expr, AxonExprParen):
        inner_expr, tp, arity = _infer_expr(
            expr.inner,
            env=env,
            expr_defs=expr_defs,
            ctx=ctx,
            module_name=module_name,
            recursive_env=recursive_env,
            expected_arity=expected_arity,
        )
        return _annotate_expr(replace(expr, inner=inner_expr), tp, arity=arity, ctx=ctx), tp, arity
    if isinstance(expr, AxonExprAscribe):
        inner_expr, inner_tp, _ = _infer_expr(
            expr.expr,
            env=env,
            expr_defs=expr_defs,
            ctx=ctx,
            module_name=module_name,
            recursive_env=recursive_env,
        )
        _unify(
            inner_tp,
            _scoped_typevars(
                expr.type_expr, module_name=module_name, ctx=ctx, freshen_generics=False
            ),
            ctx,
        )
        tp = _scoped_typevars(
            expr.type_expr, module_name=module_name, ctx=ctx, freshen_generics=False
        )
        return _annotate_expr(replace(expr, expr=inner_expr), tp, arity=1, ctx=ctx), tp, 1
    if isinstance(expr, AxonExprList):
        typed_items: list[AxonExpr] = []
        item_type: TypeExpr | None = None
        for item in expr.items:
            typed_item, item_tp, _ = _infer_expr(
                item,
                env=env,
                expr_defs=expr_defs,
                ctx=ctx,
                module_name=module_name,
                recursive_env=recursive_env,
            )
            typed_items.append(typed_item)
            item_type = item_tp if item_type is None else _unify(item_type, item_tp, ctx)
        if item_type is not None:
            typed_items = [_retag_numeric_literals(item, item_type, ctx) for item in typed_items]
        list_type = TypeList(item=item_type or TypeAny())
        return (
            _annotate_expr(replace(expr, items=tuple(typed_items)), list_type, arity=1, ctx=ctx),
            list_type,
            1,
        )
    if isinstance(expr, AxonExprTuple):
        typed_tuple_items: list[AxonExpr] = []
        item_types: list[TypeExpr] = []
        for item in expr.items:
            typed_item, item_tp, _ = _infer_expr(
                item,
                env=env,
                expr_defs=expr_defs,
                ctx=ctx,
                module_name=module_name,
                recursive_env=recursive_env,
            )
            typed_tuple_items.append(typed_item)
            item_types.append(item_tp)
        tuple_type = TypeTuple(items=tuple(item_types))
        return (
            _annotate_expr(
                replace(expr, items=tuple(typed_tuple_items)),
                tuple_type,
                arity=len(item_types),
                ctx=ctx,
            ),
            tuple_type,
            len(item_types),
        )
    if isinstance(expr, AxonExprBinary):
        left_expr, left_tp, _ = _infer_expr(
            expr.left,
            env=env,
            expr_defs=expr_defs,
            ctx=ctx,
            module_name=module_name,
            recursive_env=recursive_env,
        )
        right_expr, right_tp, _ = _infer_expr(
            expr.right,
            env=env,
            expr_defs=expr_defs,
            ctx=ctx,
            module_name=module_name,
            recursive_env=recursive_env,
        )
        if expr.op in _COMPARE_OPS | _BOOL_OPS:
            if expr.op in _COMPARE_OPS:
                left_expanded = _expand_alias(_apply_subst(left_tp, ctx), ctx)
                right_expanded = _expand_alias(_apply_subst(right_tp, ctx), ctx)
                if isinstance(left_expanded, TypeTensor) and isinstance(
                    right_expanded, TypeTensor
                ):
                    dims = _unify_broadcast_tensor_dims(
                        tuple(_normalize_dim_token(dim, ctx) for dim in left_expanded.dims),
                        tuple(_normalize_dim_token(dim, ctx) for dim in right_expanded.dims),
                        ctx,
                    )
                    tp = TypeTensor(
                        base="Tensor",
                        dims=tuple(_normalize_dim_token(dim, ctx) for dim in dims),
                    ) if dims is not None else _unify(left_tp, right_tp, ctx)
                elif isinstance(left_expanded, TypeTensor) and _is_scalar_numeric_type(
                    right_expanded
                ):
                    tp = _apply_subst(left_tp, ctx)
                elif isinstance(right_expanded, TypeTensor) and _is_scalar_numeric_type(
                    left_expanded
                ):
                    tp = _apply_subst(right_tp, ctx)
                else:
                    _unify(left_tp, right_tp, ctx)
                    tp = TypeBool()
            else:
                _unify(left_tp, TypeBool(), ctx)
                _unify(right_tp, TypeBool(), ctx)
                tp = TypeBool()
        elif expr.op in _ARITH_OPS:
            left_expanded = _expand_alias(_apply_subst(left_tp, ctx), ctx)
            right_expanded = _expand_alias(_apply_subst(right_tp, ctx), ctx)
            if isinstance(left_expanded, TypeTensor) and isinstance(right_expanded, TypeTensor):
                dims = _unify_broadcast_tensor_dims(
                    tuple(_normalize_dim_token(dim, ctx) for dim in left_expanded.dims),
                    tuple(_normalize_dim_token(dim, ctx) for dim in right_expanded.dims),
                    ctx,
                )
                if dims is None:
                    tp = _unify(left_tp, right_tp, ctx)
                else:
                    tp = TypeTensor(
                        base="Tensor",
                        dims=tuple(_normalize_dim_token(dim, ctx) for dim in dims),
                    )
            elif isinstance(left_expanded, TypeTensor) and _is_scalar_numeric_type(right_expanded):
                tp = _apply_subst(left_tp, ctx)
            elif isinstance(right_expanded, TypeTensor) and _is_scalar_numeric_type(left_expanded):
                tp = _apply_subst(right_tp, ctx)
            elif isinstance(left_expanded, TypeTensor):
                tp = _apply_subst(left_tp, ctx)
            elif isinstance(right_expanded, TypeTensor):
                tp = _apply_subst(right_tp, ctx)
            elif isinstance(left_expanded, TypeVar | TypeAny) and _is_scalar_numeric_type(
                right_expanded
            ):
                tp = _apply_subst(left_tp, ctx)
            elif isinstance(right_expanded, TypeVar | TypeAny) and _is_scalar_numeric_type(
                left_expanded
            ):
                tp = _apply_subst(right_tp, ctx)
            elif isinstance(left_expanded, TypeDim) or isinstance(right_expanded, TypeDim):
                tp = TypeDim()
                _unify(left_tp, tp, ctx)
                _unify(right_tp, tp, ctx)
            elif isinstance(left_expanded, TypeVar | TypeAny) and isinstance(
                right_expanded, TypeFloat | TypeInt
            ):
                tp = _apply_subst(left_tp, ctx)
            elif isinstance(right_expanded, TypeVar | TypeAny) and isinstance(
                left_expanded, TypeFloat | TypeInt
            ):
                tp = _apply_subst(right_tp, ctx)
            elif isinstance(left_expanded, TypeVar | TypeAny) and isinstance(
                right_expanded, TypeVar | TypeAny
            ):
                tp = _apply_subst(left_tp, ctx)
            elif isinstance(left_expanded, TypeFloat) or isinstance(right_expanded, TypeFloat):
                tp = TypeFloat()
                _unify(left_tp, tp, ctx)
                _unify(right_tp, tp, ctx)
            elif isinstance(left_expanded, TypeInt) and isinstance(right_expanded, TypeInt):
                _unify(left_tp, TypeInt(), ctx)
                _unify(right_tp, TypeInt(), ctx)
                tp = TypeInt()
            else:
                tp = ctx.fresh_type_var()
                _unify(left_tp, tp, ctx)
                _unify(right_tp, tp, ctx)
            left_expr = _retag_numeric_literals(left_expr, tp, ctx)
            right_expr = _retag_numeric_literals(right_expr, tp, ctx)
        else:
            tp = TypeAny()
        typed = replace(expr, left=left_expr, right=right_expr)
        return _annotate_expr(typed, tp, arity=1, ctx=ctx), tp, 1
    if isinstance(expr, AxonExprCall):
        typed_args: list[AxonExpr] = []
        arg_types: list[TypeExpr] = []
        for arg in expr.args:
            typed_arg, arg_tp, _ = _infer_expr(
                arg,
                env=env,
                expr_defs=expr_defs,
                ctx=ctx,
                module_name=module_name,
                recursive_env=recursive_env,
            )
            typed_args.append(typed_arg)
            arg_types.append(arg_tp)
        typed_kwargs: dict[str, AxonKwargValue] = {}
        kwarg_types: dict[str, TypeExpr] = {}
        for key, raw_value in expr.kwargs.items():
            if isinstance(raw_value, AxonExpr):
                typed_value, value_tp, _ = _infer_expr(
                    raw_value,
                    env=env,
                    expr_defs=expr_defs,
                    ctx=ctx,
                    module_name=module_name,
                    recursive_env=recursive_env,
                )
                typed_kwargs[key] = typed_value
                kwarg_types[key] = value_tp
            else:
                typed_kwargs[key] = raw_value
        callee = ctx.modules_by_name.get(expr.callee)
        if callee is None:
            primitive = _infer_primitive_call(
                callee=expr.callee,
                typed_args=typed_args,
                arg_types=arg_types,
                typed_kwargs=typed_kwargs,
                kwarg_types=kwarg_types,
                ctx=ctx,
                module_name=module_name,
                expected_arity=expected_arity,
                expr_defs=expr_defs,
            )
            if primitive is not None:
                return primitive
            missing_callee_result_type: TypeExpr = ctx.fresh_type_var()
            typed_call = replace(expr, args=tuple(typed_args), kwargs=typed_kwargs)
            return (
                _annotate_expr(
                    typed_call, missing_callee_result_type, arity=expected_arity or 1, ctx=ctx
                ),
                missing_callee_result_type,
                expected_arity or 1,
            )

        if recursive_env is not None and expr.callee in recursive_env.signatures:
            param_types, return_types = recursive_env.signatures[expr.callee]
            dim_instantiation_map: dict[str, DimToken] = {}
        else:
            param_types, return_types = _instantiate_module_signature(callee, ctx)
            dim_instantiation_map = _module_dim_instantiation_map(callee, return_types)
        param_names = [*callee.path_params, *(param.name for param in callee.params)]
        implicit_path_count = _implicit_leading_path_param_count(callee, arg_types, ctx)
        positional_param_types = param_types[implicit_path_count:]
        dim_subst: dict[str, DimToken | tuple[DimToken, ...]] = {}
        positional_count = len(typed_args)
        for idx, (arg_tp, param_tp) in enumerate(
            zip(arg_types, positional_param_types[:positional_count], strict=False)
        ):
            param_type_idx = implicit_path_count + idx
            try:
                _unify(arg_tp, param_tp, ctx)
            except ValueError as exc:
                raise ValueError(
                    f"{expr.callee} positional arg {idx + 1}: {exc}; "
                    f"actual={arg_tp!r} expected={param_tp!r}"
                ) from exc
            typed_args[idx] = _retag_numeric_literals(typed_args[idx], param_tp, ctx)
            if param_type_idx >= len(callee.path_params):
                param_idx = param_type_idx - len(callee.path_params)
                if 0 <= param_idx < len(callee.params):
                    _collect_module_call_dim_subst(
                        expected_type=param_types[param_type_idx],
                        actual_type=arg_tp,
                        actual_expr=typed_args[idx],
                        ctx=ctx,
                        subst=dim_subst,
                    )
                    raw_param_type = callee.params[param_idx].type_expr
                    dim_expr = typed_args[idx]
                    if isinstance(raw_param_type, TypeDim) or (
                        isinstance(raw_param_type, TypeOptional)
                        and isinstance(raw_param_type.inner, TypeDim)
                    ):
                        dim_token = _expr_to_dim_token_resolved(dim_expr, expr_defs)
                        if dim_token is not None:
                            dim_subst.setdefault(callee.params[param_idx].name, dim_token)
                            if isinstance(dim_expr, AxonExprName):
                                dim_subst[dim_expr.name] = dim_token
                            instantiated_dim = dim_instantiation_map.get(
                                callee.params[param_idx].name
                            )
                            if isinstance(instantiated_dim, str):
                                dim_subst.setdefault(instantiated_dim, dim_token)
        for key, value_tp in kwarg_types.items():
            if key in param_names:
                param_idx = param_names.index(key)
                if param_idx < len(param_types):
                    try:
                        _unify(value_tp, param_types[param_idx], ctx)
                    except ValueError as exc:
                        raise ValueError(f"{expr.callee} kwarg {key}: {exc}") from exc
                    typed_kwarg_expr = typed_kwargs.get(key)
                    if isinstance(typed_kwarg_expr, AxonExpr):
                        typed_kwargs[key] = _retag_numeric_literals(
                            typed_kwarg_expr, param_types[param_idx], ctx
                        )
                    if param_idx >= len(callee.path_params):
                        raw_param_idx = param_idx - len(callee.path_params)
                        if 0 <= raw_param_idx < len(callee.params):
                            raw_value = typed_kwargs.get(key)
                            if isinstance(raw_value, AxonExpr):
                                _collect_module_call_dim_subst(
                                    expected_type=param_types[param_idx],
                                    actual_type=value_tp,
                                    actual_expr=raw_value,
                                    ctx=ctx,
                                    subst=dim_subst,
                                )
                                raw_param_type = callee.params[raw_param_idx].type_expr
                                if isinstance(raw_param_type, TypeDim) or (
                                    isinstance(raw_param_type, TypeOptional)
                                    and isinstance(raw_param_type.inner, TypeDim)
                                ):
                                    dim_token = _expr_to_dim_token_resolved(raw_value, expr_defs)
                                    if dim_token is not None:
                                        dim_subst.setdefault(
                                            callee.params[raw_param_idx].name, dim_token
                                        )
                                        if isinstance(raw_value, AxonExprName):
                                            dim_subst[raw_value.name] = dim_token
                                        instantiated_dim = dim_instantiation_map.get(
                                            callee.params[raw_param_idx].name
                                        )
                                        if isinstance(instantiated_dim, str):
                                            dim_subst.setdefault(instantiated_dim, dim_token)
        if dim_subst:
            return_types = [_substitute_type_dims(tp, subst=dim_subst) for tp in return_types]
        refined_body_result = _refined_definition_call_result_type(
            callee=callee,
            typed_args=tuple(typed_args),
            arg_types=tuple(arg_types),
            typed_kwargs=typed_kwargs,
            kwarg_types=kwarg_types,
            ctx=ctx,
            module_name=module_name,
            recursive_env=recursive_env,
            expected_arity=expected_arity,
        )
        if refined_body_result is not None:
            if dim_subst:
                refined_body_result = _substitute_type_dims(
                    refined_body_result, subst=dim_subst
                )
            if len(return_types) == 1:
                if _type_specificity_score(refined_body_result) < _type_specificity_score(
                    return_types[0]
                ):
                    return_types = [_apply_subst(refined_body_result, ctx)]
            elif isinstance(refined_body_result, TypeTuple) and len(
                refined_body_result.items
            ) == len(return_types):
                refined_items: list[TypeExpr] = []
                for refined_item, return_item in zip(
                    refined_body_result.items, return_types, strict=True
                ):
                    if _type_specificity_score(refined_item) < _type_specificity_score(
                        return_item
                    ):
                        refined_items.append(_apply_subst(refined_item, ctx))
                    else:
                        refined_items.append(return_item)
                return_types = refined_items
        if len(return_types) == 1:
            call_result_type: TypeExpr = _resolve_type_dim_aliases(
                _apply_subst(return_types[0], ctx), expr_defs
            )
            arity = (
                expected_arity
                if isinstance(_expand_alias(call_result_type, ctx), TypeList) and expected_arity
                else 1
            )
        else:
            call_result_type = _resolve_type_dim_aliases(
                TypeTuple(items=tuple(_apply_subst(tp, ctx) for tp in return_types)),
                expr_defs,
            )
            arity = len(return_types)
        typed_call = replace(expr, args=tuple(typed_args), kwargs=typed_kwargs)
        return (
            _annotate_expr(typed_call, call_result_type, arity=arity, ctx=ctx),
            call_result_type,
            arity,
        )
    if isinstance(expr, AxonExprLambda):
        body_env = dict(env)
        body_env[expr.var] = ctx.fresh_type_var()
        body_expr_defs = dict(expr_defs)
        body_expr_defs.pop(expr.var, None)
        typed_body, body_tp, _ = _infer_expr(
            expr.body,
            env=body_env,
            expr_defs=body_expr_defs,
            ctx=ctx,
            module_name=module_name,
            recursive_env=recursive_env,
        )
        lambda_type = ctx.fresh_type_var()
        return (
            _annotate_expr(replace(expr, body=typed_body), lambda_type, arity=1, ctx=ctx),
            lambda_type,
            1,
        )
    if isinstance(expr, AxonExprIf | AxonExprTernary):
        cond_expr, cond_tp, _ = _infer_expr(
            expr.cond,
            env=env,
            expr_defs=expr_defs,
            ctx=ctx,
            module_name=module_name,
            recursive_env=recursive_env,
        )
        _unify(cond_tp, TypeBool(), ctx)
        true_expr, true_tp, _ = _infer_expr(
            expr.true_expr,
            env=env,
            expr_defs=expr_defs,
            ctx=ctx,
            module_name=module_name,
            recursive_env=recursive_env,
            expected_arity=expected_arity,
        )
        false_expr, false_tp, _ = _infer_expr(
            expr.false_expr,
            env=env,
            expr_defs=expr_defs,
            ctx=ctx,
            module_name=module_name,
            recursive_env=recursive_env,
            expected_arity=expected_arity,
        )
        result_tp = _join_branch_types(true_tp, false_tp, ctx)
        typed_cond_expr = replace(expr, cond=cond_expr, true_expr=true_expr, false_expr=false_expr)
        return (
            _annotate_expr(typed_cond_expr, result_tp, arity=expected_arity or 1, ctx=ctx),
            result_tp,
            expected_arity or 1,
        )
    if isinstance(expr, AxonExprDo):
        typed_do_body, _, body_return_types = _infer_statements(
            expr.body,
            env=dict(env),
            expr_defs=dict(expr_defs),
            condition_defs={},
            ctx=ctx,
            module_name=module_name,
            recursive_env=recursive_env,
            expected_return_types=None,
        )
        if not body_return_types:
            do_result_tp: TypeExpr = TypeAny()
            arity = 1
        elif len(body_return_types[0]) == 1:
            do_result_tp = body_return_types[0][0]
            arity = 1
        else:
            tuple_items = body_return_types[0]
            do_result_tp = TypeTuple(items=tuple_items)
            arity = len(tuple_items)
        return (
            _annotate_expr(replace(expr, body=typed_do_body), do_result_tp, arity=arity, ctx=ctx),
            do_result_tp,
            arity,
        )
    if isinstance(expr, AxonExprPipe):
        raise ValueError("Axon typecheck failed: pipe should not remain in flat AST")
    result_tp = TypeAny()
    return _annotate_expr(expr, result_tp, arity=1, ctx=ctx), result_tp, 1


def _merge_branch_envs(
    base_env: dict[str, TypeExpr],
    left_env: dict[str, TypeExpr],
    right_env: dict[str, TypeExpr],
    ctx: _TcCtx,
) -> dict[str, TypeExpr]:
    merged = dict(base_env)
    for name in set(left_env) & set(right_env):
        try:
            merged[name] = _unify(left_env[name], right_env[name], ctx)
        except ValueError:
            base_tp = base_env.get(name)
            if base_tp is None:
                raise
            try:
                _unify(left_env[name], base_tp, ctx)
                _unify(right_env[name], base_tp, ctx)
            except ValueError:
                raise
            merged[name] = _apply_subst(base_tp, ctx)
    return merged


def _refine_env_for_condition(
    cond: AxonExpr,
    *,
    env: dict[str, TypeExpr],
    ctx: _TcCtx,
    truthy: bool,
) -> dict[str, TypeExpr]:
    refined = dict(env)
    if not isinstance(cond, AxonExprBinary) or cond.op not in {"==", "!="}:
        return refined
    if isinstance(cond.left, AxonExprName) and isinstance(cond.right, AxonExprNull):
        name = cond.left.name
    elif isinstance(cond.right, AxonExprName) and isinstance(cond.left, AxonExprNull):
        name = cond.right.name
    else:
        return refined
    current = refined.get(name)
    if current is None:
        return refined
    current = _apply_subst(current, ctx)
    matches_null = truthy if cond.op == "==" else not truthy
    if matches_null:
        refined[name] = TypeNull()
        return refined
    expanded = _expand_alias(current, ctx)
    if isinstance(expanded, TypeOptional):
        refined[name] = expanded.inner
    return refined


def _infer_scope_body_return(
    statements: tuple[AxonStatement, ...],
    *,
    env: dict[str, TypeExpr],
    expr_defs: dict[str, AxonExpr],
    condition_defs: dict[str, AxonExpr],
    ctx: _TcCtx,
    module_name: str,
    recursive_env: _RecursiveInterfaces | None,
    expected_arity: int | None,
) -> tuple[tuple[AxonStatement, ...], tuple[TypeExpr, ...]]:
    typed_body, _, return_types = _infer_statements(
        statements,
        env=dict(env),
        expr_defs=dict(expr_defs),
        condition_defs=dict(condition_defs),
        ctx=ctx,
        module_name=module_name,
        recursive_env=recursive_env,
        expected_return_types=None,
        in_loop=False,
    )
    if not return_types:
        if expected_arity is None:
            return typed_body, (TypeAny(),)
        return typed_body, tuple(ctx.fresh_type_var() for _ in range(expected_arity))
    target = return_types[0]
    for branch in return_types[1:]:
        if len(branch) != len(target):
            raise ValueError("Axon typecheck failed: inconsistent scope return arity")
        target = tuple(_unify(a, b, ctx) for a, b in zip(target, branch, strict=True))
    return typed_body, target


def _infer_statements(
    statements: tuple[AxonStatement, ...],
    *,
    env: dict[str, TypeExpr],
    expr_defs: dict[str, AxonExpr],
    condition_defs: dict[str, AxonExpr],
    ctx: _TcCtx,
    module_name: str,
    recursive_env: _RecursiveInterfaces | None,
    expected_return_types: tuple[TypeExpr, ...] | None,
    in_loop: bool = False,
) -> tuple[tuple[AxonStatement, ...], dict[str, TypeExpr], list[tuple[TypeExpr, ...]]]:
    typed: list[AxonStatement] = []
    current_env = dict(env)
    current_expr_defs = dict(expr_defs)
    current_condition_defs = dict(condition_defs)
    returns: list[tuple[TypeExpr, ...]] = []
    for stmt in statements:
        if isinstance(stmt, AxonBind):
            typed_expr, expr_tp, expr_arity = _infer_expr(
                stmt.expr,
                env=current_env,
                expr_defs=current_expr_defs,
                ctx=ctx,
                module_name=module_name,
                recursive_env=recursive_env,
                expected_arity=len(stmt.targets) if len(stmt.targets) > 1 else None,
            )
            target_types = _destructure_type(expr_tp, len(stmt.targets), ctx)
            for name, tp in zip(stmt.targets, target_types, strict=True):
                if name != "_":
                    # Axon binds shadow prior names after the RHS has been inferred.
                    # Rebinding is common for shape-changing tensor transforms such as
                    # `x <- reshape x ...`; requiring the new type to unify with the
                    # old one incorrectly rejects valid programs.
                    tp = _apply_subst(tp, ctx)
                    current_env[name] = tp
                    if len(stmt.targets) == 1:
                        current_expr_defs[name] = typed_expr
                    else:
                        current_expr_defs.pop(name, None)
                    if (
                        len(stmt.targets) == 1
                        and isinstance(stmt.expr, AxonExprBinary)
                        and stmt.expr.op in {"==", "!="}
                    ):
                        current_condition_defs[name] = stmt.expr
                    else:
                        current_condition_defs.pop(name, None)
                else:
                    current_expr_defs.pop(name, None)
            typed.append(replace(stmt, expr=typed_expr))
            continue
        if isinstance(stmt, AxonReturn | AxonYield):
            if in_loop and isinstance(stmt, AxonReturn):
                raise ValueError(
                    "Axon typecheck failed: return is not valid inside for-loop bodies; use yield"
                )
            typed_values: list[AxonExpr] = []
            value_types: list[TypeExpr] = []
            for value in stmt.values:
                typed_value, value_tp, _ = _infer_expr(
                    value,
                    env=current_env,
                    expr_defs=current_expr_defs,
                    ctx=ctx,
                    module_name=module_name,
                    recursive_env=recursive_env,
                )
                typed_values.append(typed_value)
                value_types.append(value_tp)
            if expected_return_types is not None:
                if len(value_types) == 1 and len(expected_return_types) > 1:
                    value_types = list(
                        _destructure_type(value_types[0], len(expected_return_types), ctx)
                    )
                if len(expected_return_types) != len(value_types):
                    raise ValueError("Axon typecheck failed: return arity mismatch")
                unified_value_types: list[TypeExpr] = []
                for actual, expected in zip(value_types, expected_return_types, strict=True):
                    try:
                        unified_value_types.append(_unify(actual, expected, ctx))
                    except ValueError as exc:
                        raise ValueError(
                            f"return type mismatch: actual={actual!r} expected={expected!r}; {exc}"
                        ) from exc
                value_types = unified_value_types
                if len(typed_values) == len(value_types):
                    typed_values = [
                        _retag_numeric_literals(value, expected, ctx)
                        for value, expected in zip(typed_values, value_types, strict=True)
                    ]
            returns.append(tuple(value_types))
            typed.append(replace(stmt, values=tuple(typed_values)))
            continue
        if isinstance(stmt, AxonCond):
            typed_cond, cond_tp, _ = _infer_expr(
                stmt.cond,
                env=current_env,
                expr_defs=current_expr_defs,
                ctx=ctx,
                module_name=module_name,
                recursive_env=recursive_env,
            )
            _unify(cond_tp, TypeBool(), ctx)
            refine_expr = (
                current_condition_defs.get(stmt.cond.name, stmt.cond)
                if isinstance(stmt.cond, AxonExprName)
                else stmt.cond
            )
            typed_true, env_true, returns_true = _infer_statements(
                stmt.true_body,
                env=_refine_env_for_condition(refine_expr, env=current_env, ctx=ctx, truthy=True),
                expr_defs=dict(current_expr_defs),
                condition_defs=dict(current_condition_defs),
                ctx=ctx,
                module_name=module_name,
                recursive_env=recursive_env,
                expected_return_types=expected_return_types,
                in_loop=in_loop,
            )
            typed_false, env_false, returns_false = _infer_statements(
                stmt.false_body,
                env=_refine_env_for_condition(refine_expr, env=current_env, ctx=ctx, truthy=False),
                expr_defs=dict(current_expr_defs),
                condition_defs=dict(current_condition_defs),
                ctx=ctx,
                module_name=module_name,
                recursive_env=recursive_env,
                expected_return_types=expected_return_types,
                in_loop=in_loop,
            )
            current_env = _merge_branch_envs(current_env, env_true, env_false, ctx)
            returns.extend(returns_true)
            returns.extend(returns_false)
            typed.append(
                replace(stmt, cond=typed_cond, true_body=typed_true, false_body=typed_false)
            )
            continue
        if isinstance(stmt, AxonScopeBind):
            typed_body, scope_return_types = _infer_scope_body_return(
                stmt.body,
                env=current_env,
                expr_defs=current_expr_defs,
                condition_defs=current_condition_defs,
                ctx=ctx,
                module_name=module_name,
                recursive_env=recursive_env,
                expected_arity=len(stmt.targets) if len(stmt.targets) > 1 else 1,
            )
            for name, tp in zip(stmt.targets, scope_return_types, strict=True):
                if name != "_":
                    current_env[name] = tp
                    current_expr_defs.pop(name, None)
            typed_kwargs: dict[str, AxonKwargValue] = {}
            for key, raw_value in stmt.kwargs.items():
                if isinstance(raw_value, AxonExpr):
                    typed_value, _, _ = _infer_expr(
                        raw_value,
                        env=current_env,
                        expr_defs=current_expr_defs,
                        ctx=ctx,
                        module_name=module_name,
                        recursive_env=recursive_env,
                    )
                    typed_kwargs[key] = typed_value
                else:
                    typed_kwargs[key] = raw_value
            typed.append(replace(stmt, body=typed_body, kwargs=typed_kwargs))
            continue
        if isinstance(stmt, AxonRepeat):
            typed_from, from_tp, _ = _infer_expr(
                stmt.from_expr,
                env=current_env,
                expr_defs=current_expr_defs,
                ctx=ctx,
                module_name=module_name,
                recursive_env=recursive_env,
            )
            typed_to, to_tp, _ = _infer_expr(
                stmt.to_expr,
                env=current_env,
                expr_defs=current_expr_defs,
                ctx=ctx,
                module_name=module_name,
                recursive_env=recursive_env,
            )
            typed_step, step_tp, _ = _infer_expr(
                stmt.step_expr,
                env=current_env,
                expr_defs=current_expr_defs,
                ctx=ctx,
                module_name=module_name,
                recursive_env=recursive_env,
            )
            _unify(from_tp, TypeInt(), ctx)
            _unify(to_tp, TypeInt(), ctx)
            _unify(step_tp, TypeInt(), ctx)
            loop_env = dict(current_env)
            loop_env[stmt.var] = TypeInt()
            carry_names = tuple(stmt.carry or ())
            carry_types: list[TypeExpr] = []
            for name in carry_names:
                carry_types.append(loop_env.get(name, ctx.fresh_type_var()))
                loop_env[name] = carry_types[-1]
            typed_body, _, loop_returns = _infer_statements(
                stmt.body,
                env=loop_env,
                expr_defs={},
                condition_defs={},
                ctx=ctx,
                module_name=module_name,
                recursive_env=recursive_env,
                expected_return_types=tuple(carry_types) if carry_types else None,
                in_loop=True,
            )
            if carry_names and stmt.targets is not None:
                for name, tp in zip(stmt.targets, carry_types, strict=True):
                    if name != "_":
                        current_env[name] = tp
                        current_expr_defs.pop(name, None)
            returns.extend(loop_returns)
            typed.append(
                replace(
                    stmt,
                    from_expr=typed_from,
                    to_expr=typed_to,
                    step_expr=typed_step,
                    body=typed_body,
                )
            )
            continue
        typed.append(stmt)
    return tuple(typed), current_env, returns


def _normalize_type_expr(tp: TypeExpr | None, ctx: _TcCtx) -> TypeExpr | None:
    if tp is None:
        return None
    tp = _apply_subst(tp, ctx)
    if isinstance(tp, TypeOptional):
        inner = _normalize_type_expr(tp.inner, ctx)
        assert inner is not None
        return TypeOptional(inner=inner)
    if isinstance(tp, TypeList):
        item = _normalize_type_expr(tp.item, ctx)
        assert item is not None
        return TypeList(item=item)
    if isinstance(tp, TypeTuple):
        return TypeTuple(items=tuple(cast(TypeExpr, _normalize_type_expr(item, ctx)) for item in tp.items))
    if isinstance(tp, TypeTensor):
        return TypeTensor(
            base=tp.base,
            dims=tuple(_normalize_dim_token(dim, ctx) for dim in tp.dims),
        )
    if isinstance(tp, TypeNamed):
        return TypeNamed(
            name=tp.name,
            args=tuple(_normalize_dim_token(dim, ctx) for dim in tp.args),
        )
    return tp


def _is_generated_dim_name(name: str) -> bool:
    inner = name[2:] if name.startswith("..") else name
    return (
        inner.startswith("__d")
        or inner.startswith("__gdim")
        or inner.startswith("__tc")
        or inner.startswith("_T")
    )


def _preferred_dim_roots_for_module(
    ctx: _TcCtx,
    header_names: set[str],
    statement_names: set[str],
) -> dict[DimToken, str]:
    if not header_names and not statement_names:
        return {}
    by_root: dict[str, list[str]] = {}
    for name in header_names | statement_names:
        normalized = _normalize_dim_token(name, ctx)
        if isinstance(normalized, str):
            by_root.setdefault(normalized, []).append(name)
    out: dict[DimToken, str] = {}
    for root, members in by_root.items():
        header_members = [name for name in members if name in header_names]
        readable_header_members = [
            name for name in header_members if not _is_generated_dim_name(name)
        ]
        if readable_header_members:
            if len(set(readable_header_members)) > 1:
                continue
            chosen = max(readable_header_members, key=_dim_name_priority)
        elif header_members:
            readable_statement_members = [
                name
                for name in members
                if name in statement_names and not _is_generated_dim_name(name)
            ]
            chosen = (
                max(readable_statement_members, key=_dim_name_priority)
                if readable_statement_members
                else max(header_members, key=_dim_name_priority)
            )
        else:
            continue
        if root != chosen:
            out[root] = chosen
        for name in header_members:
            if name != chosen:
                out[name] = chosen
        for name in header_members:
            if _is_generated_dim_name(name):
                continue
            normalized = _normalize_dim_token(name, ctx)
            if normalized != name:
                out[normalized] = chosen
    for name in header_names:
        if _is_generated_dim_name(name):
            continue
        normalized = _normalize_dim_token(name, ctx)
        if normalized != name:
            out[normalized] = name
    return out


def _rewrite_preferred_dim_token(
    dim: DimToken, preferred_dim_roots: dict[DimToken, str] | None
) -> DimToken:
    if not preferred_dim_roots:
        return dim
    replacement = preferred_dim_roots.get(dim)
    if replacement is not None:
        return replacement
    if isinstance(dim, str):
        return preferred_dim_roots.get(dim, dim)
    if isinstance(dim, int):
        return dim
    return DimExprBinary(
        op=dim.op,
        left=_rewrite_preferred_dim_token(dim.left, preferred_dim_roots),
        right=_rewrite_preferred_dim_token(dim.right, preferred_dim_roots),
    )


def _normalize_dim_token_for_module(
    dim: DimToken,
    ctx: _TcCtx,
    preferred_dim_roots: dict[DimToken, str] | None = None,
) -> DimToken:
    return _rewrite_preferred_dim_token(_normalize_dim_token(dim, ctx), preferred_dim_roots)


def _apply_preferred_dim_names_to_type_expr(
    tp: TypeExpr | None,
    preferred_dim_roots: dict[DimToken, str] | None = None,
) -> TypeExpr | None:
    if tp is None or not preferred_dim_roots:
        return tp
    if isinstance(tp, TypeOptional):
        inner = _apply_preferred_dim_names_to_type_expr(tp.inner, preferred_dim_roots)
        assert inner is not None
        return TypeOptional(inner=inner)
    if isinstance(tp, TypeList):
        item = _apply_preferred_dim_names_to_type_expr(tp.item, preferred_dim_roots)
        assert item is not None
        return TypeList(item=item)
    if isinstance(tp, TypeTuple):
        return TypeTuple(
            items=tuple(
                item
                for item in (
                    _apply_preferred_dim_names_to_type_expr(item, preferred_dim_roots)
                    for item in tp.items
                )
                if item is not None
            )
        )
    if isinstance(tp, TypeTensor):
        return TypeTensor(
            base=tp.base,
            dims=tuple(_rewrite_preferred_dim_token(dim, preferred_dim_roots) for dim in tp.dims),
        )
    if isinstance(tp, TypeNamed):
        return TypeNamed(
            name=tp.name,
            args=tuple(_rewrite_preferred_dim_token(dim, preferred_dim_roots) for dim in tp.args),
        )
    return tp


def _unqualify_type_vars(tp: TypeExpr | None) -> TypeExpr | None:
    if tp is None:
        return None
    if isinstance(tp, TypeVar):
        return TypeVar(name=tp.name.rsplit("::", 1)[-1])
    if isinstance(tp, TypeOptional):
        inner = _unqualify_type_vars(tp.inner)
        assert inner is not None
        return TypeOptional(inner=inner)
    if isinstance(tp, TypeList):
        item = _unqualify_type_vars(tp.item)
        assert item is not None
        return TypeList(item=item)
    if isinstance(tp, TypeTuple):
        return TypeTuple(items=tuple(cast(TypeExpr, _unqualify_type_vars(item)) for item in tp.items))
    if isinstance(tp, TypeTensor):
        return TypeTensor(base=tp.base, dims=tp.dims)
    if isinstance(tp, TypeNamed):
        return TypeNamed(name=tp.name, args=tp.args)
    return tp


def _normalize_type_expr_for_module(
    tp: TypeExpr | None,
    ctx: _TcCtx,
    preferred_dim_roots: dict[DimToken, str] | None = None,
) -> TypeExpr | None:
    return _unqualify_type_vars(
        _apply_preferred_dim_names_to_type_expr(
            _normalize_type_expr(tp, ctx), preferred_dim_roots
        )
    )


def _apply_type_subst_only(tp: TypeExpr, ctx: _TcCtx) -> TypeExpr:
    if isinstance(tp, TypeVar):
        current: TypeExpr = tp
        seen: set[str] = set()
        while (
            isinstance(current, TypeVar)
            and current.name in ctx.substitutions
            and current.name not in seen
        ):
            seen.add(current.name)
            next_tp = ctx.substitutions[current.name]
            if next_tp == current:
                break
            current = next_tp
        if current != tp:
            return _apply_type_subst_only(current, ctx)
        return current
    if isinstance(tp, TypeOptional):
        return TypeOptional(inner=_apply_type_subst_only(tp.inner, ctx))
    if isinstance(tp, TypeList):
        return TypeList(item=_apply_type_subst_only(tp.item, ctx))
    if isinstance(tp, TypeTuple):
        return TypeTuple(items=tuple(_apply_type_subst_only(item, ctx) for item in tp.items))
    return tp


def _replace_expr_annotations(
    expr: AxonExpr,
    *,
    inferred_type: TypeExpr | None,
    inferred_arity: int | None,
    inferred_dims: tuple[DimToken, ...] | None,
) -> AxonExpr:
    return replace(
        expr,
        inferred_type=inferred_type,
        inferred_arity=inferred_arity,
        inferred_dims=inferred_dims,
    )


def _normalize_expr(
    expr: AxonExpr,
    ctx: _TcCtx,
    *,
    preferred_dim_roots: dict[DimToken, str] | None = None,
) -> AxonExpr:
    inferred_type = _normalize_type_expr_for_module(expr.inferred_type, ctx, preferred_dim_roots)
    inferred_arity = expr.inferred_arity
    if isinstance(inferred_type, TypeTuple):
        inferred_arity = len(inferred_type.items)
    inferred_dims = (
        tuple(
            _normalize_dim_token_for_module(dim, ctx, preferred_dim_roots)
            for dim in expr.inferred_dims
        )
        if expr.inferred_dims
        else None
    )
    if inferred_dims is None and inferred_type is not None:
        inferred_dims = _type_dims(inferred_type, ctx)
    if isinstance(expr, AxonExprName) and expr.name.startswith("__d"):
        mapped = ctx.dim_substitutions.get(expr.name)
        if not isinstance(mapped, tuple) and mapped is not None:
            normalized_mapped = _normalize_dim_token(mapped, ctx)
            if isinstance(normalized_mapped, int):
                return _replace_expr_annotations(
                    AxonExprInt(value=normalized_mapped),
                    inferred_type=inferred_type,
                    inferred_arity=inferred_arity,
                    inferred_dims=inferred_dims,
                )
            if isinstance(normalized_mapped, str) and normalized_mapped != expr.name:
                return _replace_expr_annotations(
                    AxonExprName(name=normalized_mapped),
                    inferred_type=inferred_type,
                    inferred_arity=inferred_arity,
                    inferred_dims=inferred_dims,
                )
    if isinstance(expr, AxonExprList):
        return _replace_expr_annotations(
            replace(
                expr,
                items=tuple(
                    _normalize_expr(item, ctx, preferred_dim_roots=preferred_dim_roots)
                    for item in expr.items
                ),
            ),
            inferred_type=inferred_type,
            inferred_arity=inferred_arity,
            inferred_dims=inferred_dims,
        )
    if isinstance(expr, AxonExprTuple):
        return _replace_expr_annotations(
            replace(
                expr,
                items=tuple(
                    _normalize_expr(item, ctx, preferred_dim_roots=preferred_dim_roots)
                    for item in expr.items
                ),
            ),
            inferred_type=inferred_type,
            inferred_arity=inferred_arity,
            inferred_dims=inferred_dims,
        )
    if isinstance(expr, AxonExprCall):
        kwargs: dict[str, AxonKwargValue] = {}
        for key, raw_value in expr.kwargs.items():
            kwargs[key] = (
                _normalize_expr(raw_value, ctx, preferred_dim_roots=preferred_dim_roots)
                if isinstance(raw_value, AxonExpr)
                else raw_value
            )
        return _replace_expr_annotations(
            replace(
                expr,
                args=tuple(
                    _normalize_expr(arg, ctx, preferred_dim_roots=preferred_dim_roots)
                    for arg in expr.args
                ),
                kwargs=kwargs,
            ),
            inferred_type=inferred_type,
            inferred_arity=inferred_arity,
            inferred_dims=inferred_dims,
        )
    if isinstance(expr, AxonExprPipe):
        return _replace_expr_annotations(
            replace(
                expr,
                value=_normalize_expr(expr.value, ctx, preferred_dim_roots=preferred_dim_roots),
                stages=tuple(
                    _normalize_expr(stage, ctx, preferred_dim_roots=preferred_dim_roots)
                    for stage in expr.stages
                ),
            ),
            inferred_type=inferred_type,
            inferred_arity=inferred_arity,
            inferred_dims=inferred_dims,
        )
    if isinstance(expr, AxonExprBind):
        return _replace_expr_annotations(
            replace(
                expr,
                value=_normalize_expr(expr.value, ctx, preferred_dim_roots=preferred_dim_roots),
                body=_normalize_expr(expr.body, ctx, preferred_dim_roots=preferred_dim_roots),
            ),
            inferred_type=inferred_type,
            inferred_arity=inferred_arity,
            inferred_dims=inferred_dims,
        )
    if isinstance(expr, AxonExprIf | AxonExprTernary):
        return _replace_expr_annotations(
            replace(
                expr,
                cond=_normalize_expr(expr.cond, ctx, preferred_dim_roots=preferred_dim_roots),
                true_expr=_normalize_expr(
                    expr.true_expr, ctx, preferred_dim_roots=preferred_dim_roots
                ),
                false_expr=_normalize_expr(
                    expr.false_expr, ctx, preferred_dim_roots=preferred_dim_roots
                ),
            ),
            inferred_type=inferred_type,
            inferred_arity=inferred_arity,
            inferred_dims=inferred_dims,
        )
    if isinstance(expr, AxonExprBinary):
        return _replace_expr_annotations(
            replace(
                expr,
                left=_normalize_expr(expr.left, ctx, preferred_dim_roots=preferred_dim_roots),
                right=_normalize_expr(expr.right, ctx, preferred_dim_roots=preferred_dim_roots),
            ),
            inferred_type=inferred_type,
            inferred_arity=inferred_arity,
            inferred_dims=inferred_dims,
        )
    if isinstance(expr, AxonExprLambda):
        return _replace_expr_annotations(
            replace(
                expr, body=_normalize_expr(expr.body, ctx, preferred_dim_roots=preferred_dim_roots)
            ),
            inferred_type=inferred_type,
            inferred_arity=inferred_arity,
            inferred_dims=inferred_dims,
        )
    if isinstance(expr, AxonExprParen):
        inner = _normalize_expr(expr.inner, ctx, preferred_dim_roots=preferred_dim_roots)
        return _replace_expr_annotations(
            inner,
            inferred_type=inferred_type,
            inferred_arity=inferred_arity,
            inferred_dims=inferred_dims,
        )
    if isinstance(expr, AxonExprAscribe):
        normalized_type_expr = _normalize_type_expr_for_module(
            expr.type_expr, ctx, preferred_dim_roots
        )
        assert normalized_type_expr is not None
        return _replace_expr_annotations(
            replace(
                expr,
                expr=_normalize_expr(expr.expr, ctx, preferred_dim_roots=preferred_dim_roots),
                type_expr=normalized_type_expr,
            ),
            inferred_type=inferred_type,
            inferred_arity=inferred_arity,
            inferred_dims=inferred_dims,
        )
    if isinstance(expr, AxonExprDo):
        return _replace_expr_annotations(
            replace(
                expr,
                body=tuple(
                    _normalize_statement(stmt, ctx, preferred_dim_roots=preferred_dim_roots)
                    for stmt in expr.body
                ),
            ),
            inferred_type=inferred_type,
            inferred_arity=inferred_arity,
            inferred_dims=inferred_dims,
        )
    return _replace_expr_annotations(
        expr,
        inferred_type=inferred_type,
        inferred_arity=inferred_arity,
        inferred_dims=inferred_dims,
    )


def _normalize_statement(
    stmt: AxonStatement,
    ctx: _TcCtx,
    *,
    preferred_dim_roots: dict[DimToken, str] | None = None,
) -> AxonStatement:
    if isinstance(stmt, AxonBind):
        return replace(
            stmt, expr=_normalize_expr(stmt.expr, ctx, preferred_dim_roots=preferred_dim_roots)
        )
    if isinstance(stmt, AxonReturn | AxonYield):
        return replace(
            stmt,
            values=tuple(
                _normalize_expr(value, ctx, preferred_dim_roots=preferred_dim_roots)
                for value in stmt.values
            ),
        )
    if isinstance(stmt, AxonCond):
        return replace(
            stmt,
            cond=_normalize_expr(stmt.cond, ctx, preferred_dim_roots=preferred_dim_roots),
            true_body=tuple(
                _normalize_statement(item, ctx, preferred_dim_roots=preferred_dim_roots)
                for item in stmt.true_body
            ),
            false_body=tuple(
                _normalize_statement(item, ctx, preferred_dim_roots=preferred_dim_roots)
                for item in stmt.false_body
            ),
        )
    if isinstance(stmt, AxonScopeBind):
        kwargs: dict[str, AxonKwargValue] = {}
        for key, raw_value in stmt.kwargs.items():
            kwargs[key] = (
                _normalize_expr(raw_value, ctx, preferred_dim_roots=preferred_dim_roots)
                if isinstance(raw_value, AxonExpr)
                else raw_value
            )
        return replace(
            stmt,
            body=tuple(
                _normalize_statement(item, ctx, preferred_dim_roots=preferred_dim_roots)
                for item in stmt.body
            ),
            kwargs=kwargs,
        )
    if isinstance(stmt, AxonRepeat):
        return replace(
            stmt,
            from_expr=_normalize_expr(stmt.from_expr, ctx, preferred_dim_roots=preferred_dim_roots),
            to_expr=_normalize_expr(stmt.to_expr, ctx, preferred_dim_roots=preferred_dim_roots),
            step_expr=_normalize_expr(stmt.step_expr, ctx, preferred_dim_roots=preferred_dim_roots),
            body=tuple(
                _normalize_statement(item, ctx, preferred_dim_roots=preferred_dim_roots)
                for item in stmt.body
            ),
        )
    return stmt


def _substitute_expr_dim_names(
    expr: AxonExpr, *, subst: Mapping[str, DimToken | tuple[DimToken, ...]]
) -> AxonExpr:
    def sub_type(tp: TypeExpr | None) -> TypeExpr | None:
        return _substitute_type_dims(tp, subst=subst) if tp is not None else None

    def sub_dims(dims: tuple[DimToken, ...] | None) -> tuple[DimToken, ...] | None:
        if dims is None:
            return None
        replaced: list[DimToken] = []
        for dim in dims:
            if isinstance(dim, str) and dim in subst:
                mapped = subst[dim]
                if isinstance(mapped, tuple):
                    replaced.extend(mapped)
                else:
                    replaced.append(mapped)
                continue
            substituted = _substitute_type_dims(TypeTensor(base="Tensor", dims=(dim,)), subst=subst)
            replaced.extend(cast(TypeTensor, substituted).dims)
        return tuple(replaced)

    def retag(updated: AxonExpr) -> AxonExpr:
        return _replace_expr_annotations(
            updated,
            inferred_type=sub_type(expr.inferred_type),
            inferred_arity=expr.inferred_arity,
            inferred_dims=sub_dims(expr.inferred_dims),
        )

    if isinstance(expr, AxonExprName):
        mapped = subst.get(expr.name)
        if isinstance(mapped, str):
            return retag(replace(expr, name=mapped))
        if isinstance(mapped, int):
            return retag(AxonExprInt(value=mapped))
        return retag(expr)
    if isinstance(expr, AxonExprCall):
        kwargs: dict[str, AxonKwargValue] = {}
        for key, raw_value in expr.kwargs.items():
            kwargs[key] = (
                _substitute_expr_dim_names(raw_value, subst=subst)
                if isinstance(raw_value, AxonExpr)
                else raw_value
            )
        return retag(
            replace(
                expr,
                args=tuple(_substitute_expr_dim_names(arg, subst=subst) for arg in expr.args),
                kwargs=kwargs,
            )
        )
    if isinstance(expr, AxonExprList | AxonExprTuple):
        return retag(
            replace(
                expr,
                items=tuple(_substitute_expr_dim_names(item, subst=subst) for item in expr.items),
            )
        )
    if isinstance(expr, AxonExprBinary):
        return retag(
            replace(
                expr,
                left=_substitute_expr_dim_names(expr.left, subst=subst),
                right=_substitute_expr_dim_names(expr.right, subst=subst),
            )
        )
    if isinstance(expr, AxonExprIf | AxonExprTernary):
        return retag(
            replace(
                expr,
                cond=_substitute_expr_dim_names(expr.cond, subst=subst),
                true_expr=_substitute_expr_dim_names(expr.true_expr, subst=subst),
                false_expr=_substitute_expr_dim_names(expr.false_expr, subst=subst),
            )
        )
    if isinstance(expr, AxonExprAscribe):
        return retag(
            replace(
                expr,
                expr=_substitute_expr_dim_names(expr.expr, subst=subst),
                type_expr=_substitute_type_dims(expr.type_expr, subst=subst),
            )
        )
    if isinstance(expr, AxonExprParen):
        return retag(replace(expr, inner=_substitute_expr_dim_names(expr.inner, subst=subst)))
    if isinstance(expr, AxonExprBind):
        return retag(
            replace(
                expr,
                value=_substitute_expr_dim_names(expr.value, subst=subst),
                body=_substitute_expr_dim_names(expr.body, subst=subst),
            )
        )
    if isinstance(expr, AxonExprLambda):
        return retag(replace(expr, body=_substitute_expr_dim_names(expr.body, subst=subst)))
    if isinstance(expr, AxonExprDo):
        return retag(
            replace(
                expr,
                body=tuple(_substitute_statement_dim_names(stmt, subst=subst) for stmt in expr.body),
            )
        )
    return retag(expr)


def _substitute_statement_dim_names(
    stmt: AxonStatement, *, subst: Mapping[str, DimToken | tuple[DimToken, ...]]
) -> AxonStatement:
    if isinstance(stmt, AxonBind):
        return replace(stmt, expr=_substitute_expr_dim_names(stmt.expr, subst=subst))
    if isinstance(stmt, AxonReturn | AxonYield):
        return replace(
            stmt,
            values=tuple(_substitute_expr_dim_names(value, subst=subst) for value in stmt.values),
        )
    if isinstance(stmt, AxonCond):
        return replace(
            stmt,
            cond=_substitute_expr_dim_names(stmt.cond, subst=subst),
            true_body=tuple(_substitute_statement_dim_names(item, subst=subst) for item in stmt.true_body),
            false_body=tuple(_substitute_statement_dim_names(item, subst=subst) for item in stmt.false_body),
        )
    if isinstance(stmt, AxonRepeat):
        return replace(
            stmt,
            from_expr=_substitute_expr_dim_names(stmt.from_expr, subst=subst),
            to_expr=_substitute_expr_dim_names(stmt.to_expr, subst=subst),
            step_expr=_substitute_expr_dim_names(stmt.step_expr, subst=subst),
            body=tuple(_substitute_statement_dim_names(item, subst=subst) for item in stmt.body),
        )
    if isinstance(stmt, AxonScopeBind):
        return replace(
            stmt,
            kwargs={
                key: _substitute_expr_dim_names(value, subst=subst)
                if isinstance(value, AxonExpr)
                else value
                for key, value in stmt.kwargs.items()
            },
            body=tuple(_substitute_statement_dim_names(item, subst=subst) for item in stmt.body),
        )
    return stmt


def _normalize_module(
    module: AxonDefinition,
    *,
    raw_params: tuple[AxonParam, ...],
    refined_return: TypeExpr | None,
    ctx: _TcCtx,
    normalize_header_dims: bool,
) -> AxonDefinition:
    preserve_explicit_signature = not _is_generated_helper(module.name)
    normalized_statements = tuple(_normalize_statement(stmt, ctx) for stmt in module.statements)
    header_dim_preferences = _collect_header_dim_preferences_from_param_uses(
        raw_params, normalized_statements
    )
    if header_dim_preferences:
        normalized_statements = tuple(
            _substitute_statement_dim_names(stmt, subst=header_dim_preferences)
            for stmt in normalized_statements
        )
    param_type_refinements = _collect_param_type_refinements_from_uses(
        raw_params, normalized_statements, ctx
    )
    preferred_dim_roots = _preferred_dim_roots_for_module(
        ctx,
        _preferred_module_dim_names(raw_params, refined_return),
        _collect_statement_dim_names(normalized_statements),
    )
    if preferred_dim_roots:
        normalized_statements = tuple(
            _normalize_statement(stmt, ctx, preferred_dim_roots=preferred_dim_roots)
            for stmt in normalized_statements
        )

    def _inferred_param_type(param: AxonParam) -> TypeExpr:
        refinement = param_type_refinements.get(param.name)
        inferred = (
            _more_specific_type(param.type_expr, refinement) or TypeAny()
            if _is_generated_helper(module.name)
            else refinement or param.type_expr or TypeAny()
        )
        normalized = (
            _normalize_type_expr_for_module(inferred, ctx, preferred_dim_roots)
            if normalize_header_dims
            else _apply_type_subst_only(inferred, ctx)
        )
        return cast(
            TypeExpr,
            _apply_preferred_dim_names_to_type_expr(
                _substitute_type_dims(normalized or TypeAny(), subst=header_dim_preferences),
                preferred_dim_roots,
            ),
        )

    normalized_param_types = tuple(
        param.type_expr
        if preserve_explicit_signature and param.type_expr is not None
        else _inferred_param_type(param)
        for param in raw_params
    )
    if preserve_explicit_signature and module.return_type_expr is not None:
        normalized_return_type = module.return_type_expr
    elif refined_return is not None:
        normalized = (
            _normalize_type_expr_for_module(refined_return, ctx, preferred_dim_roots)
            if normalize_header_dims
            else _apply_type_subst_only(refined_return, ctx)
        )
        normalized_return_type = cast(
            TypeExpr,
            _apply_preferred_dim_names_to_type_expr(
                _substitute_type_dims(normalized or TypeAny(), subst=header_dim_preferences),
                preferred_dim_roots,
            ),
        )
    else:
        normalized_return_type = None
    if _is_generated_helper(module.name):
        return_items: list[TypeExpr] = (
            list(normalized_return_type.items)
            if isinstance(normalized_return_type, TypeTuple)
            else [normalized_return_type or TypeAny()]
        )
        canonical_params, canonical_returns = _canonicalize_generated_signature(
            (list(normalized_param_types), return_items)
        )
        normalized_param_types = tuple(canonical_params)
        normalized_return_type = (
            TypeTuple(items=tuple(canonical_returns))
            if len(canonical_returns) != 1
            else canonical_returns[0]
        )
    signature_dim_subst: dict[str, DimToken | tuple[DimToken, ...]] = {}

    def collect_signature_subst(raw_tp: TypeExpr | None, normalized_tp: TypeExpr | None) -> None:
        if isinstance(raw_tp, TypeTensor) and isinstance(normalized_tp, TypeTensor):
            if len(raw_tp.dims) == len(normalized_tp.dims):
                for raw_dim, normalized_dim in zip(raw_tp.dims, normalized_tp.dims, strict=True):
                    if isinstance(raw_dim, str) and raw_dim != normalized_dim:
                        signature_dim_subst.setdefault(raw_dim, normalized_dim)
        if isinstance(raw_tp, TypeOptional) and isinstance(normalized_tp, TypeOptional):
            collect_signature_subst(raw_tp.inner, normalized_tp.inner)
        if isinstance(raw_tp, TypeList) and isinstance(normalized_tp, TypeList):
            collect_signature_subst(raw_tp.item, normalized_tp.item)
        if isinstance(raw_tp, TypeTuple) and isinstance(normalized_tp, TypeTuple):
            for raw_item, normalized_item in zip(
                raw_tp.items, normalized_tp.items, strict=False
            ):
                collect_signature_subst(raw_item, normalized_item)

    for raw_param, normalized_param_type in zip(
        raw_params, normalized_param_types, strict=True
    ):
        collect_signature_subst(raw_param.type_expr, normalized_param_type)
    collect_signature_subst(refined_return, normalized_return_type)
    if signature_dim_subst:
        normalized_statements = tuple(
            _substitute_statement_dim_names(stmt, subst=signature_dim_subst)
            for stmt in normalized_statements
        )
    return replace(
        module,
        params=tuple(
            replace(
                param,
                type_expr=normalized_type,
                default_expr=(
                    _normalize_expr(param.default_expr, ctx)
                    if param.default_expr is not None
                    else None
                ),
            )
            for param, normalized_type in zip(raw_params, normalized_param_types, strict=True)
        ),
        statements=normalized_statements,
        return_type_expr=normalized_return_type,
        constraints=tuple(
            _normalize_constraint(item, ctx, preferred_dim_roots=preferred_dim_roots)
            for item in (module.constraints or ())
        ),
    )


def _typed_module_raw(
    module: AxonDefinition,
    *,
    ctx: _TcCtx,
    recursive_env: _RecursiveInterfaces | None,
) -> tuple[AxonDefinition, tuple[AxonParam, ...], TypeExpr | None, _TcCtx]:
    share_constraints = (
        recursive_env is not None
        and bool(recursive_env.members)
        and module.name in recursive_env.members
    )
    module_ctx = ctx.child(share_constraints=share_constraints)
    env: dict[str, TypeExpr] = {}
    refined_params: list = []
    shared_signature = None if recursive_env is None else recursive_env.signatures.get(module.name)
    if shared_signature is None:
        scoped_param_types = [
            _scoped_typevars(
                param.type_expr, module_name=module.name, ctx=module_ctx, freshen_generics=False
            )
            for param in module.params
        ]
        expected_returns = _module_return_types(module, module_ctx, freshen_generics=False)
    else:
        scoped_param_types = [tp for tp in shared_signature[0][len(module.path_params) :]]
        expected_returns = tuple(shared_signature[1])
    for param, inner_tp in zip(module.params, scoped_param_types, strict=True):
        param_header_type = (
            inner_tp.inner
            if param.optional and isinstance(inner_tp, TypeOptional)
            else inner_tp
        )
        env[param.name] = (
            inner_tp
            if param.optional and isinstance(inner_tp, TypeOptional)
            else TypeOptional(inner=inner_tp)
            if param.optional
            else inner_tp
        )
        refined_params.append(replace(param, type_expr=param_header_type))
    dim_bound_names = _collect_type_dim_names(module.return_type_expr)
    for param in module.params:
        dim_bound_names.update(_collect_type_dim_names(param.type_expr))
    for name in sorted(dim_bound_names):
        if name not in env:
            env[name] = TypeDim()
    for name in module.path_params:
        env[name] = TypePath()
    if module.path_param is not None:
        env[module.path_param] = TypePath()
    try:
        typed_statements, _, body_returns = _infer_statements(
            module.statements,
            env=env,
            expr_defs={},
            condition_defs={},
            ctx=module_ctx,
            module_name=module.name,
            recursive_env=recursive_env,
            expected_return_types=expected_returns,
            in_loop=False,
        )
    except ValueError as exc:
        raise ValueError(f"{module.name}: {exc}") from exc
    inferred_return = _module_return_from_body_returns(body_returns, module_ctx)
    if inferred_return is not None:
        refined_return = inferred_return
    elif shared_signature is not None:
        if len(shared_signature[1]) == 1:
            refined_return = _apply_type_subst_only(shared_signature[1][0], module_ctx)
        else:
            refined_return = TypeTuple(
                items=tuple(
                    _apply_type_subst_only(item, module_ctx) for item in shared_signature[1]
                )
            )
    elif module.return_type_expr is None:
        refined_return = None
    else:
        refined_return = _apply_type_subst_only(
            _scoped_typevars(
                module.return_type_expr,
                module_name=module.name,
                ctx=module_ctx,
                freshen_generics=False,
            ),
            module_ctx,
        )
    ctx.fresh_counter = module_ctx.fresh_counter
    raw_params = tuple(refined_params)
    raw_module = replace(
        module,
        params=raw_params,
        statements=typed_statements,
        constraints=_collect_module_constraints(
            module, statements=typed_statements, ctx=module_ctx
        ),
    )
    return raw_module, raw_params, refined_return, module_ctx


def _module_return_from_body_returns(
    body_returns: list[tuple[TypeExpr, ...]],
    ctx: _TcCtx,
) -> TypeExpr | None:
    if not body_returns:
        return None
    first = body_returns[0]
    if not first:
        return TypeTuple(items=())
    merged = tuple(_apply_subst(item, ctx) for item in first)
    for returned in body_returns[1:]:
        if len(returned) != len(merged):
            raise ValueError("Axon typecheck failed: inconsistent return arity")
        merged = tuple(
            _join_branch_types(left, right, ctx)
            for left, right in zip(merged, returned, strict=True)
        )
    if len(merged) == 1:
        return _apply_subst(merged[0], ctx)
    return TypeTuple(items=tuple(_apply_subst(item, ctx) for item in merged))


def _typed_module(
    module: AxonDefinition,
    *,
    ctx: _TcCtx,
    recursive_env: _RecursiveInterfaces | None,
) -> AxonDefinition:
    raw_module, raw_params, refined_return, module_ctx = _typed_module_raw(
        module,
        ctx=ctx,
        recursive_env=recursive_env,
    )
    return _normalize_module(
        raw_module,
        raw_params=raw_params,
        refined_return=refined_return,
        ctx=module_ctx,
        normalize_header_dims=True,
    )


def _rewrite_typed_module_headers_from_param_uses(module: AxonDefinition) -> AxonDefinition:
    header_dim_preferences = _collect_header_dim_preferences_from_param_uses(
        module.params, module.statements
    )
    if not header_dim_preferences:
        return module
    return replace(
        module,
        params=tuple(
            replace(
                param,
                type_expr=_substitute_type_dims(param.type_expr, subst=header_dim_preferences)
                if param.type_expr is not None
                else None,
            )
            for param in module.params
        ),
        return_type_expr=(
            _substitute_type_dims(module.return_type_expr, subst=header_dim_preferences)
            if module.return_type_expr is not None
            else None
        ),
    )


def _walk_typed_exprs(expr: AxonExpr):
    yield expr
    if isinstance(expr, AxonExprBinary):
        yield from _walk_typed_exprs(expr.left)
        yield from _walk_typed_exprs(expr.right)
    elif isinstance(expr, AxonExprBind):
        yield from _walk_typed_exprs(expr.value)
        yield from _walk_typed_exprs(expr.body)
    elif isinstance(expr, AxonExprCall):
        for arg in expr.args:
            yield from _walk_typed_exprs(arg)
        for value in expr.kwargs.values():
            if isinstance(value, AxonExpr):
                yield from _walk_typed_exprs(value)
    elif isinstance(expr, AxonExprDo):
        for stmt in expr.body:
            yield from _walk_typed_stmts((stmt,))
    elif isinstance(expr, AxonExprIf | AxonExprTernary):
        yield from _walk_typed_exprs(expr.cond)
        yield from _walk_typed_exprs(expr.true_expr)
        yield from _walk_typed_exprs(expr.false_expr)
    elif isinstance(expr, AxonExprLambda):
        yield from _walk_typed_exprs(expr.body)
    elif isinstance(expr, AxonExprAscribe):
        yield from _walk_typed_exprs(expr.expr)
    elif isinstance(expr, AxonExprList | AxonExprTuple):
        for item in expr.items:
            yield from _walk_typed_exprs(item)
    elif isinstance(expr, AxonExprParen):
        yield from _walk_typed_exprs(expr.inner)


def _walk_typed_stmts(statements: tuple[AxonStatement, ...]):
    for stmt in statements:
        if isinstance(stmt, AxonBind):
            yield from _walk_typed_exprs(stmt.expr)
        elif isinstance(stmt, AxonReturn | AxonYield):
            for value in stmt.values:
                yield from _walk_typed_exprs(value)
        elif isinstance(stmt, AxonCond):
            yield from _walk_typed_exprs(stmt.cond)
            yield from _walk_typed_stmts(stmt.true_body)
            yield from _walk_typed_stmts(stmt.false_body)
        elif isinstance(stmt, AxonRepeat):
            yield from _walk_typed_exprs(stmt.from_expr)
            yield from _walk_typed_exprs(stmt.to_expr)
            yield from _walk_typed_exprs(stmt.step_expr)
            yield from _walk_typed_stmts(stmt.body)
        elif isinstance(stmt, AxonScopeBind):
            for raw_value in stmt.kwargs.values():
                if isinstance(raw_value, AxonExpr):
                    yield from _walk_typed_exprs(raw_value)
            yield from _walk_typed_stmts(stmt.body)


def _walk_call_callees_expr(expr: AxonExpr):
    if isinstance(expr, AxonExprCall):
        yield expr.callee
        for arg in expr.args:
            yield from _walk_call_callees_expr(arg)
        for value in expr.kwargs.values():
            if isinstance(value, AxonExpr):
                yield from _walk_call_callees_expr(value)
        return
    if isinstance(expr, AxonExprBinary):
        yield from _walk_call_callees_expr(expr.left)
        yield from _walk_call_callees_expr(expr.right)
        return
    if isinstance(expr, AxonExprBind):
        yield from _walk_call_callees_expr(expr.value)
        yield from _walk_call_callees_expr(expr.body)
        return
    if isinstance(expr, AxonExprIf | AxonExprTernary):
        yield from _walk_call_callees_expr(expr.cond)
        yield from _walk_call_callees_expr(expr.true_expr)
        yield from _walk_call_callees_expr(expr.false_expr)
        return
    if isinstance(expr, AxonExprLambda):
        yield from _walk_call_callees_expr(expr.body)
        return
    if isinstance(expr, AxonExprAscribe):
        yield from _walk_call_callees_expr(expr.expr)
        return
    if isinstance(expr, AxonExprList | AxonExprTuple):
        for item in expr.items:
            yield from _walk_call_callees_expr(item)
        return
    if isinstance(expr, AxonExprParen):
        yield from _walk_call_callees_expr(expr.inner)
        return
    if isinstance(expr, AxonExprDo):
        yield from _walk_call_callees_stmts(expr.body)


def _walk_call_callees_stmts(statements: tuple[AxonStatement, ...]):
    for stmt in statements:
        if isinstance(stmt, AxonBind):
            yield from _walk_call_callees_expr(stmt.expr)
        elif isinstance(stmt, AxonReturn | AxonYield):
            for value in stmt.values:
                yield from _walk_call_callees_expr(value)
        elif isinstance(stmt, AxonCond):
            yield from _walk_call_callees_expr(stmt.cond)
            yield from _walk_call_callees_stmts(stmt.true_body)
            yield from _walk_call_callees_stmts(stmt.false_body)
        elif isinstance(stmt, AxonRepeat):
            yield from _walk_call_callees_expr(stmt.from_expr)
            yield from _walk_call_callees_expr(stmt.to_expr)
            yield from _walk_call_callees_expr(stmt.step_expr)
            yield from _walk_call_callees_stmts(stmt.body)
        elif isinstance(stmt, AxonScopeBind):
            for raw_value in stmt.kwargs.values():
                if isinstance(raw_value, AxonExpr):
                    yield from _walk_call_callees_expr(raw_value)
            yield from _walk_call_callees_stmts(stmt.body)


def _module_call_graph(program: AxonFile) -> dict[str, set[str]]:
    module_names = {module.name for module in program.modules}
    graph: dict[str, set[str]] = {module.name: set() for module in program.modules}
    for module in program.modules:
        for callee in _walk_call_callees_stmts(module.statements):
            if callee in module_names:
                graph[module.name].add(callee)
    return graph


def _reachable_modules(program: AxonFile, *, main_module: str | None) -> frozenset[str]:
    main_module = resolve_main_module(program, main_module=main_module)
    module_names = {module.name for module in program.modules}
    if main_module not in module_names:
        return frozenset()
    return frozenset(reachable_definitions(program, entrypoint=main_module))


def _prune_to_main(program: AxonFile, *, main_module: str | None) -> AxonFile:
    keep = _reachable_modules(program, main_module=main_module)
    if main_module is not None and main_module not in keep:
        raise ValueError(f"Axon typecheck failed: main module {main_module!r} not found")
    if len(keep) == len(program.modules):
        return program
    return replace(program, modules=tuple(module for module in program.modules if module.name in keep))


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


def _program_type_order(program: AxonFile) -> tuple[list[tuple[str, ...]], _RecursiveInterfaces]:
    graph = _module_call_graph(program)
    sccs = _module_sccs(graph)
    component_of: dict[str, int] = {}
    for idx, component in enumerate(sccs):
        for name in component:
            component_of[name] = idx
    condensed: dict[int, set[int]] = {idx: set() for idx in range(len(sccs))}
    indegree: dict[int, int] = {idx: 0 for idx in range(len(sccs))}
    for caller, callees in graph.items():
        caller_comp = component_of[caller]
        for callee in callees:
            callee_comp = component_of[callee]
            if caller_comp == callee_comp or callee_comp in condensed[caller_comp]:
                continue
            condensed[caller_comp].add(callee_comp)
            indegree[callee_comp] += 1
    queue = [idx for idx, degree in indegree.items() if degree == 0]
    topo: list[int] = []
    while queue:
        current = queue.pop()
        topo.append(current)
        for succ in condensed[current]:
            indegree[succ] -= 1
            if indegree[succ] == 0:
                queue.append(succ)
    ordered_components: list[tuple[str, ...]] = []
    recursive_members: set[str] = set()
    for comp_idx in reversed(topo):
        component = tuple(
            module.name for module in program.modules if component_of[module.name] == comp_idx
        )
        ordered_components.append(component)
        if len(component) > 1:
            recursive_members.update(component)
            continue
        only = component[0]
        if only in graph[only]:
            recursive_members.add(only)
    return ordered_components, _RecursiveInterfaces(
        signatures={}, members=frozenset(recursive_members)
    )


def _is_generated_helper(module_name: str) -> bool:
    return "__loop_" in module_name or "__cond_" in module_name


def _is_loop_helper(module_name: str) -> bool:
    return "__loop_" in module_name


def _type_specificity_score(tp: TypeExpr | None) -> int:
    def generated_dim_score(dim: DimToken) -> int:
        if isinstance(dim, str) and dim.lstrip(".").startswith("__"):
            return 5
        return 0

    if tp is None:
        return 1000
    if isinstance(tp, TypeAny):
        return 1000
    if isinstance(tp, TypeVar):
        return 100
    if isinstance(tp, TypeOptional):
        return _type_specificity_score(tp.inner)
    if isinstance(tp, TypeList):
        return _type_specificity_score(tp.item)
    if isinstance(tp, TypeTuple):
        return sum(_type_specificity_score(item) for item in tp.items)
    if isinstance(tp, TypeTensor):
        score = 0
        for dim in tp.dims:
            if isinstance(dim, str) and dim.startswith(".."):
                score += 20
            score += generated_dim_score(dim)
        return score
    if isinstance(tp, TypeNamed):
        score = 0
        for dim in tp.args:
            if isinstance(dim, str) and dim.startswith(".."):
                score += 20
            score += generated_dim_score(dim)
        return score
    return 0


def _more_specific_type(candidate: TypeExpr | None, fallback: TypeExpr | None) -> TypeExpr | None:
    if candidate is None:
        return fallback
    if fallback is None:
        return candidate
    return (
        candidate
        if _type_specificity_score(candidate) <= _type_specificity_score(fallback)
        else fallback
    )


def _module_param_type_by_name(module: AxonDefinition, name: str) -> TypeExpr | None:
    if name in module.path_params or name == module.path_param:
        return TypePath()
    for param in module.params:
        if param.name == name:
            return param.type_expr
    return None


def _collect_callsite_param_refinements(
    modules_by_name: dict[str, AxonDefinition],
    ctx: _TcCtx,
    *,
    only_generated: bool = False,
    skip_conflicts: bool = False,
) -> dict[str, dict[str, TypeExpr]]:
    refinements: dict[str, dict[str, TypeExpr]] = {}
    conflicted: set[tuple[str, str]] = set()

    def record(callee: AxonDefinition, param_name: str, tp: TypeExpr | None) -> None:
        if tp is None:
            return
        if only_generated and not _is_generated_helper(callee.name):
            return
        if (callee.name, param_name) in conflicted:
            return
        by_param = refinements.setdefault(callee.name, {})
        normalized = _apply_subst(tp, ctx)
        current = by_param.get(param_name)
        try:
            by_param[param_name] = (
                normalized if current is None else _join_branch_types(current, normalized, ctx)
            )
        except ValueError:
            if not skip_conflicts:
                raise
            conflicted.add((callee.name, param_name))
            by_param.pop(param_name, None)

    def visit_expr(expr: AxonExpr) -> None:
        if isinstance(expr, AxonExprCall):
            callee = modules_by_name.get(expr.callee)
            if callee is not None:
                param_names = [*callee.path_params, *(param.name for param in callee.params)]
                for idx, arg in enumerate(expr.args):
                    if idx >= len(param_names):
                        continue
                    record(callee, param_names[idx], arg.inferred_type)
                for key, raw_value in expr.kwargs.items():
                    if key in param_names and isinstance(raw_value, AxonExpr):
                        record(callee, key, raw_value.inferred_type)
            for arg in expr.args:
                visit_expr(arg)
            for raw_value in expr.kwargs.values():
                if isinstance(raw_value, AxonExpr):
                    visit_expr(raw_value)
            return
        if isinstance(expr, AxonExprBinary):
            visit_expr(expr.left)
            visit_expr(expr.right)
            return
        if isinstance(expr, AxonExprBind):
            visit_expr(expr.value)
            visit_expr(expr.body)
            return
        if isinstance(expr, AxonExprIf | AxonExprTernary):
            visit_expr(expr.cond)
            visit_expr(expr.true_expr)
            visit_expr(expr.false_expr)
            return
        if isinstance(expr, AxonExprLambda):
            visit_expr(expr.body)
            return
        if isinstance(expr, AxonExprAscribe):
            visit_expr(expr.expr)
            return
        if isinstance(expr, AxonExprList | AxonExprTuple):
            for item in expr.items:
                visit_expr(item)
            return
        if isinstance(expr, AxonExprParen):
            visit_expr(expr.inner)
            return
        if isinstance(expr, AxonExprPipe):
            visit_expr(expr.value)
            for stage in expr.stages:
                visit_expr(stage)
            return
        if isinstance(expr, AxonExprDo):
            for stmt in expr.body:
                visit_stmt(stmt)

    def visit_stmt(stmt: AxonStatement) -> None:
        if isinstance(stmt, AxonBind):
            visit_expr(stmt.expr)
            return
        if isinstance(stmt, AxonReturn | AxonYield):
            for value in stmt.values:
                visit_expr(value)
            return
        if isinstance(stmt, AxonCond):
            visit_expr(stmt.cond)
            for item in stmt.true_body:
                visit_stmt(item)
            for item in stmt.false_body:
                visit_stmt(item)
            return
        if isinstance(stmt, AxonRepeat):
            visit_expr(stmt.from_expr)
            visit_expr(stmt.to_expr)
            visit_expr(stmt.step_expr)
            for item in stmt.body:
                visit_stmt(item)
            return
        if isinstance(stmt, AxonScopeBind):
            for raw_value in stmt.kwargs.values():
                if isinstance(raw_value, AxonExpr):
                    visit_expr(raw_value)
            for item in stmt.body:
                visit_stmt(item)

    for module in modules_by_name.values():
        for statement in module.statements:
            visit_stmt(statement)
    return refinements


def _apply_callsite_param_refinements(
    module: AxonDefinition,
    refinements: dict[str, TypeExpr],
    ctx: _TcCtx,
    *,
    skip_conflicts: bool = False,
    only_broad: bool = False,
    require_specificity_improvement: bool = False,
) -> AxonDefinition:
    if not refinements:
        return module
    refined_params: list[AxonParam] = []
    for param in module.params:
        refined = refinements.get(param.name)
        if refined is None:
            refined_params.append(param)
            continue
        current = param.type_expr or TypeAny()
        if only_broad and not isinstance(current, TypeAny | TypeVar):
            refined_params.append(param)
            continue
        if param.optional and isinstance(refined, TypeOptional):
            refined = refined.inner
        if isinstance(refined, TypeOptional) and not isinstance(current, TypeOptional):
            refined = refined.inner
        try:
            join_ctx = _TcCtx(
                modules_by_name=ctx.modules_by_name,
                type_aliases=ctx.type_aliases,
                substitutions=dict(ctx.substitutions),
                dim_substitutions=dict(ctx.dim_substitutions),
                fresh_counter=ctx.fresh_counter,
            )
            joined = (
                _apply_subst(current, join_ctx)
                if _apply_subst(current, join_ctx) == _apply_subst(refined, join_ctx)
                else _join_branch_types(current, refined, join_ctx)
            )
            if require_specificity_improvement and _type_specificity_score(
                refined
            ) < _type_specificity_score(current):
                joined = _apply_subst(refined, join_ctx)
            ctx.fresh_counter = max(ctx.fresh_counter, join_ctx.fresh_counter)
        except ValueError:
            if skip_conflicts:
                refined_params.append(param)
                continue
            raise
        if (
            require_specificity_improvement
            and _type_specificity_score(joined) >= _type_specificity_score(current)
        ):
            refined_params.append(param)
            continue
        refined_params.append(
            replace(
                param,
                type_expr=_apply_subst(joined, join_ctx),
            )
        )
    return replace(module, params=tuple(refined_params))


def _loop_helper_return_arity(module: AxonDefinition) -> int:
    if isinstance(module.return_type_expr, TypeTuple):
        return len(module.return_type_expr.items)
    return 1


def _return_type_from_signature(return_types: list[TypeExpr]) -> TypeExpr | None:
    if not return_types:
        return None
    if len(return_types) == 1:
        return _clone_type_expr(return_types[0])
    return TypeTuple(items=tuple(_clone_type_expr(item) for item in return_types))


def _prefer_signature_interface(
    module: AxonDefinition,
    signature: tuple[list[TypeExpr], list[TypeExpr]],
) -> AxonDefinition:
    def collect_dim_subst(
        old_tp: TypeExpr | None,
        new_tp: TypeExpr | None,
        out: dict[str, DimToken | tuple[DimToken, ...]],
    ) -> None:
        if isinstance(old_tp, TypeOptional) and isinstance(new_tp, TypeOptional):
            collect_dim_subst(old_tp.inner, new_tp.inner, out)
            return
        if isinstance(old_tp, TypeList) and isinstance(new_tp, TypeList):
            collect_dim_subst(old_tp.item, new_tp.item, out)
            return
        if isinstance(old_tp, TypeTuple) and isinstance(new_tp, TypeTuple):
            for old_item, new_item in zip(old_tp.items, new_tp.items, strict=False):
                collect_dim_subst(old_item, new_item, out)
            return
        if isinstance(old_tp, TypeTensor) and isinstance(new_tp, TypeTensor):
            if len(old_tp.dims) != len(new_tp.dims):
                return
            for old_dim, new_dim in zip(old_tp.dims, new_tp.dims, strict=True):
                if isinstance(old_dim, str) and old_dim != new_dim:
                    out.setdefault(old_dim, new_dim)
            return
        if isinstance(old_tp, TypeNamed) and isinstance(new_tp, TypeNamed):
            if old_tp.name != new_tp.name or len(old_tp.args) != len(new_tp.args):
                return
            for old_dim, new_dim in zip(old_tp.args, new_tp.args, strict=True):
                if isinstance(old_dim, str) and old_dim != new_dim:
                    out.setdefault(old_dim, new_dim)

    param_types, return_types = signature
    value_param_types = param_types[len(module.path_params) :]
    params: list[AxonParam] = []
    dim_subst: dict[str, DimToken | tuple[DimToken, ...]] = {}
    for param, signature_type in zip(module.params, value_param_types, strict=False):
        preferred = _more_specific_type(param.type_expr, signature_type) or param.type_expr
        collect_dim_subst(param.type_expr, preferred, dim_subst)
        params.append(
            replace(param, type_expr=_clone_type_expr(preferred) if preferred is not None else None)
        )
    if len(params) < len(module.params):
        params.extend(module.params[len(params) :])
    signature_return = _return_type_from_signature(return_types)
    preferred_return = _more_specific_type(module.return_type_expr, signature_return)
    collect_dim_subst(module.return_type_expr, preferred_return, dim_subst)
    return replace(
        module,
        params=tuple(params),
        statements=tuple(
            _substitute_statement_dim_names(statement, subst=dim_subst)
            for statement in module.statements
        )
        if dim_subst
        else module.statements,
        return_type_expr=(
            _clone_type_expr(preferred_return) if preferred_return is not None else None
        ),
    )


def _build_recursive_interfaces(
    program: AxonFile, ctx: _TcCtx, recursive_members: frozenset[str]
) -> _RecursiveInterfaces:
    signatures: dict[str, tuple[list[TypeExpr], list[TypeExpr]]] = {}
    for module in program.modules:
        if module.name not in recursive_members:
            continue
        signatures[module.name] = _clone_signature(_instantiate_module_signature(module, ctx))
    return _RecursiveInterfaces(signatures=signatures, members=recursive_members)


def _strip_inferred_expr(expr: AxonExpr) -> AxonExpr:
    expr = replace(expr, inferred_type=None, inferred_arity=None, inferred_dims=None)
    if isinstance(expr, AxonExprBinary):
        return replace(expr, left=_strip_inferred_expr(expr.left), right=_strip_inferred_expr(expr.right))
    if isinstance(expr, AxonExprBind):
        return replace(
            expr,
            value=_strip_inferred_expr(expr.value),
            body=_strip_inferred_expr(expr.body),
        )
    if isinstance(expr, AxonExprCall):
        return replace(
            expr,
            args=tuple(_strip_inferred_expr(arg) for arg in expr.args),
            kwargs={
                key: _strip_inferred_expr(value) if isinstance(value, AxonExpr) else value
                for key, value in expr.kwargs.items()
            },
        )
    if isinstance(expr, AxonExprDo):
        return replace(expr, body=tuple(_strip_inferred_stmt(stmt) for stmt in expr.body))
    if isinstance(expr, AxonExprIf | AxonExprTernary):
        return replace(
            expr,
            cond=_strip_inferred_expr(expr.cond),
            true_expr=_strip_inferred_expr(expr.true_expr),
            false_expr=_strip_inferred_expr(expr.false_expr),
        )
    if isinstance(expr, AxonExprLambda):
        return replace(expr, body=_strip_inferred_expr(expr.body))
    if isinstance(expr, AxonExprAscribe):
        return replace(expr, expr=_strip_inferred_expr(expr.expr))
    if isinstance(expr, AxonExprList | AxonExprTuple):
        return replace(expr, items=tuple(_strip_inferred_expr(item) for item in expr.items))
    if isinstance(expr, AxonExprParen):
        return replace(expr, inner=_strip_inferred_expr(expr.inner))
    if isinstance(expr, AxonExprPipe):
        return replace(
            expr,
            value=_strip_inferred_expr(expr.value),
            stages=tuple(_strip_inferred_expr(stage) for stage in expr.stages),
        )
    return expr


def _strip_inferred_stmt(stmt: AxonStatement) -> AxonStatement:
    if isinstance(stmt, AxonBind):
        return replace(stmt, expr=_strip_inferred_expr(stmt.expr))
    if isinstance(stmt, AxonReturn | AxonYield):
        return replace(stmt, values=tuple(_strip_inferred_expr(value) for value in stmt.values))
    if isinstance(stmt, AxonCond):
        return replace(
            stmt,
            cond=_strip_inferred_expr(stmt.cond),
            true_body=tuple(_strip_inferred_stmt(item) for item in stmt.true_body),
            false_body=tuple(_strip_inferred_stmt(item) for item in stmt.false_body),
        )
    if isinstance(stmt, AxonRepeat):
        return replace(
            stmt,
            from_expr=_strip_inferred_expr(stmt.from_expr),
            to_expr=_strip_inferred_expr(stmt.to_expr),
            step_expr=_strip_inferred_expr(stmt.step_expr),
            body=tuple(_strip_inferred_stmt(item) for item in stmt.body),
        )
    if isinstance(stmt, AxonScopeBind):
        prefix = _strip_inferred_expr(stmt.prefix)
        return replace(
            stmt,
            prefix=cast(AxonExprPath, prefix),
            body=tuple(_strip_inferred_stmt(item) for item in stmt.body),
            kwargs={
                key: _strip_inferred_expr(value) if isinstance(value, AxonExpr) else value
                for key, value in stmt.kwargs.items()
            },
        )
    return stmt


def _strip_inferred_program(program: AxonFile) -> AxonFile:
    return replace(
        program,
        modules=tuple(
            replace(
                module,
                statements=tuple(_strip_inferred_stmt(stmt) for stmt in module.statements),
                body_expr=(
                    _strip_inferred_expr(module.body_expr)
                    if module.body_expr is not None
                    else None
                ),
            )
            for module in program.modules
        ),
    )


def _typecheck_recursive_component(
    component: tuple[str, ...],
    *,
    modules_by_name: dict[str, AxonDefinition],
    ctx: _TcCtx,
    recursive_env: _RecursiveInterfaces,
) -> dict[str, AxonDefinition]:
    component_signatures = {
        name: _clone_signature(recursive_env.signatures[name]) for name in component
    }
    if all(
        modules_by_name[name].return_type_expr is not None
        and all(param.type_expr is not None for param in modules_by_name[name].params)
        and not _is_generated_helper(name)
        for name in component
    ):
        typed_group: dict[str, AxonDefinition] = {}
        for name in component:
            raw_module, raw_params, refined_return, module_ctx = _typed_module_raw(
                modules_by_name[name],
                ctx=ctx,
                recursive_env=recursive_env,
            )
            typed_group[name] = _normalize_module(
                raw_module,
                raw_params=raw_params,
                refined_return=refined_return,
                ctx=module_ctx,
                normalize_header_dims=True,
            )
            recursive_env.signatures[name] = _clone_signature(
                _signature_from_module_header(typed_group[name])
            )
        return typed_group
    final_group: dict[str, AxonDefinition] = {}
    max_iterations = 12
    for _ in range(max_iterations):
        iter_ctx = _TcCtx(
            modules_by_name=ctx.modules_by_name,
            type_aliases=ctx.type_aliases,
            substitutions={},
            dim_substitutions={},
            fresh_counter=ctx.fresh_counter,
        )
        iter_recursive_signatures = {
            **{
                name: _clone_signature(sig)
                for name, sig in recursive_env.signatures.items()
                if name not in component_signatures
            },
            **{name: _clone_signature(sig) for name, sig in component_signatures.items()},
        }
        iter_recursive_env = _RecursiveInterfaces(
            signatures=iter_recursive_signatures, members=recursive_env.members
        )
        raw_group: list[tuple[str, AxonDefinition, tuple[AxonParam, ...], TypeExpr | None, _TcCtx]] = []
        normalized_group: dict[str, AxonDefinition] = {}
        for name in component:
            raw_module, raw_params, refined_return, module_ctx = _typed_module_raw(
                modules_by_name[name],
                ctx=iter_ctx,
                recursive_env=iter_recursive_env,
            )
            raw_group.append((name, raw_module, raw_params, refined_return, module_ctx))
        for name, raw_module, raw_params, refined_return, module_ctx in raw_group:
            normalized_group[name] = _normalize_module(
                raw_module,
                raw_params=raw_params,
                refined_return=refined_return,
                ctx=module_ctx,
                normalize_header_dims=True,
            )
        callsite_refinements = (
            {}
            if all(_is_loop_helper(name) for name in component)
            else _collect_callsite_param_refinements(normalized_group, iter_ctx)
        )
        if callsite_refinements:
            normalized_group = {
                name: _apply_callsite_param_refinements(
                    module,
                    callsite_refinements.get(name, {}),
                    iter_ctx,
                )
                for name, module in normalized_group.items()
            }
        forced_group = {
            name: _prefer_signature_interface(normalized_group[name], component_signatures[name])
            if _is_generated_helper(name)
            else normalized_group[name]
            for name in component
        }
        forced_changed = any(
            not ast_equal(forced_group[name], normalized_group[name]) for name in component
        )
        normalized_group = forced_group
        new_signatures = {
            name: _canonicalize_generated_signature(
                _signature_from_module_header(normalized_group[name])
            )
            for name in component
        }
        old_signatures = {
            name: _canonicalize_generated_signature(component_signatures[name])
            for name in component
        }
        ctx.fresh_counter = max(
            ctx.fresh_counter,
            iter_ctx.fresh_counter,
            *(module_ctx.fresh_counter for _, _, _, _, module_ctx in raw_group),
        )
        previous_group = final_group
        final_group = normalized_group
        if all(new_signatures[name] == old_signatures[name] for name in component):
            for name in component:
                recursive_env.signatures[name] = _clone_signature(
                    _signature_from_module_header(normalized_group[name])
                )
            return final_group
        if forced_changed:
            if previous_group and all(
                ast_equal(normalized_group[name], previous_group[name]) for name in component
            ):
                for name in component:
                    recursive_env.signatures[name] = _clone_signature(
                        _signature_from_module_header(normalized_group[name])
                    )
                return final_group
            component_signatures = {
                name: _clone_signature(_signature_from_module_header(normalized_group[name]))
                for name in component
            }
            continue
        component_signatures = {
            name: _clone_signature(_signature_from_module_header(normalized_group[name]))
            for name in component
        }
    raise ValueError(
        f"Axon typecheck failed: recursive signature refinement did not converge for {component!r}"
    )


def _typecheck_flat_once(program: AxonFile) -> AxonFile:
    program = _strip_inferred_program(program)
    ctx = _TcCtx(
        modules_by_name={module.name: module for module in program.modules},
        type_aliases=dict(program.type_aliases),
        substitutions={},
        dim_substitutions={},
    )
    ordered_components, recursive_stub = _program_type_order(program)
    recursive_env = _build_recursive_interfaces(program, ctx, recursive_stub.members)
    modules_by_name = {module.name: module for module in program.modules}
    typed_modules_by_name: dict[str, AxonDefinition] = {}
    for component in ordered_components:
        if len(component) > 1 or (len(component) == 1 and component[0] in recursive_env.members):
            typed_component = _typecheck_recursive_component(
                component,
                modules_by_name=modules_by_name,
                ctx=ctx,
                recursive_env=recursive_env,
            )
            typed_modules_by_name.update(typed_component)
            ctx.modules_by_name.update(typed_component)
            continue
        name = component[0]
        typed_modules_by_name[name] = _typed_module(
            modules_by_name[name],
            ctx=ctx,
            recursive_env=recursive_env,
        )
        ctx.modules_by_name[name] = typed_modules_by_name[name]
    typed_modules = tuple(
        _rewrite_typed_module_headers_from_param_uses(typed_modules_by_name[module.name])
        for module in program.modules
    )
    typed_program = AxonFile(
        modules=typed_modules,
        imports=program.imports,
        imported_members=dict(program.imported_members),
        exports=program.exports,
        pragmas=dict(program.pragmas),
        type_aliases=dict(program.type_aliases),
        origin_path=program.origin_path,
    )
    return _thread_interprocedural_call_guards(typed_program)


def _apply_global_generated_callsite_refinements(program: AxonFile) -> tuple[AxonFile, bool]:
    ctx = _TcCtx(
        modules_by_name={module.name: module for module in program.modules},
        type_aliases=dict(program.type_aliases),
        substitutions={},
        dim_substitutions={},
    )
    modules_by_name = {module.name: module for module in program.modules}
    callsite_refinements = _collect_callsite_param_refinements(
        modules_by_name,
        ctx,
        only_generated=True,
        skip_conflicts=True,
    )
    if not callsite_refinements:
        return program, False
    changed = False
    refined_modules: list[AxonDefinition] = []
    for module in program.modules:
        if not _is_generated_helper(module.name) or _is_loop_helper(module.name):
            refined_modules.append(module)
            continue
        refined = _apply_callsite_param_refinements(
            module,
            callsite_refinements.get(module.name, {}),
            ctx,
            skip_conflicts=True,
            require_specificity_improvement=True,
        )
        changed = changed or not ast_equal(refined, module)
        refined_modules.append(refined)
    if not changed:
        return program, False
    return replace(program, modules=tuple(refined_modules)), True


@dataclass(frozen=True)
class _CallsiteTypes:
    callee: str
    arg_types: tuple[TypeExpr | None, ...]
    kwarg_types: Mapping[str, TypeExpr | None]
    result_type: TypeExpr | None


def _collect_external_component_callsites(
    program: AxonFile, component: frozenset[str], *, external_only: bool = True
) -> list[_CallsiteTypes]:
    out: list[_CallsiteTypes] = []

    def visit_expr(expr: AxonExpr, *, external: bool) -> None:
        if isinstance(expr, AxonExprCall):
            if external and expr.callee in component:
                out.append(
                    _CallsiteTypes(
                        callee=expr.callee,
                        arg_types=tuple(arg.inferred_type for arg in expr.args),
                        kwarg_types={
                            key: value.inferred_type if isinstance(value, AxonExpr) else None
                            for key, value in expr.kwargs.items()
                        },
                        result_type=expr.inferred_type,
                    )
                )
            for arg in expr.args:
                visit_expr(arg, external=external)
            for raw_value in expr.kwargs.values():
                if isinstance(raw_value, AxonExpr):
                    visit_expr(raw_value, external=external)
            return
        if isinstance(expr, AxonExprBinary):
            visit_expr(expr.left, external=external)
            visit_expr(expr.right, external=external)
            return
        if isinstance(expr, AxonExprBind):
            visit_expr(expr.value, external=external)
            visit_expr(expr.body, external=external)
            return
        if isinstance(expr, AxonExprIf | AxonExprTernary):
            visit_expr(expr.cond, external=external)
            visit_expr(expr.true_expr, external=external)
            visit_expr(expr.false_expr, external=external)
            return
        if isinstance(expr, AxonExprLambda):
            visit_expr(expr.body, external=external)
            return
        if isinstance(expr, AxonExprAscribe):
            visit_expr(expr.expr, external=external)
            return
        if isinstance(expr, AxonExprList | AxonExprTuple):
            for item in expr.items:
                visit_expr(item, external=external)
            return
        if isinstance(expr, AxonExprParen):
            visit_expr(expr.inner, external=external)
            return
        if isinstance(expr, AxonExprPipe):
            visit_expr(expr.value, external=external)
            for stage in expr.stages:
                visit_expr(stage, external=external)
            return
        if isinstance(expr, AxonExprDo):
            for stmt in expr.body:
                visit_stmt(stmt, external=external)

    def visit_stmt(stmt: AxonStatement, *, external: bool) -> None:
        if isinstance(stmt, AxonBind):
            visit_expr(stmt.expr, external=external)
            return
        if isinstance(stmt, AxonReturn | AxonYield):
            for value in stmt.values:
                visit_expr(value, external=external)
            return
        if isinstance(stmt, AxonCond):
            visit_expr(stmt.cond, external=external)
            for item in stmt.true_body:
                visit_stmt(item, external=external)
            for item in stmt.false_body:
                visit_stmt(item, external=external)
            return
        if isinstance(stmt, AxonRepeat):
            visit_expr(stmt.from_expr, external=external)
            visit_expr(stmt.to_expr, external=external)
            visit_expr(stmt.step_expr, external=external)
            for item in stmt.body:
                visit_stmt(item, external=external)
            return
        if isinstance(stmt, AxonScopeBind):
            for raw_value in stmt.kwargs.values():
                if isinstance(raw_value, AxonExpr):
                    visit_expr(raw_value, external=external)
            for item in stmt.body:
                visit_stmt(item, external=external)

    for module in program.modules:
        external = module.name not in component or not external_only
        for statement in module.statements:
            visit_stmt(statement, external=external)
    return out


def _return_type_from_callsite_result(tp: TypeExpr | None) -> TypeExpr | None:
    return _clone_type_expr(tp) if tp is not None else None


def _specialize_module_from_callsite(
    module: AxonDefinition, callsite: _CallsiteTypes
) -> AxonDefinition:
    dim_subst: dict[str, DimToken | tuple[DimToken, ...]] = {}

    def collect_dim_subst(old_tp: TypeExpr | None, new_tp: TypeExpr | None) -> None:
        if isinstance(old_tp, TypeOptional) and isinstance(new_tp, TypeOptional):
            collect_dim_subst(old_tp.inner, new_tp.inner)
            return
        if isinstance(old_tp, TypeOptional):
            collect_dim_subst(old_tp.inner, new_tp)
            return
        if isinstance(new_tp, TypeOptional):
            collect_dim_subst(old_tp, new_tp.inner)
            return
        if isinstance(old_tp, TypeList) and isinstance(new_tp, TypeList):
            collect_dim_subst(old_tp.item, new_tp.item)
            return
        if isinstance(old_tp, TypeTuple) and isinstance(new_tp, TypeTuple):
            for old_item, new_item in zip(old_tp.items, new_tp.items, strict=False):
                collect_dim_subst(old_item, new_item)
            return
        if isinstance(old_tp, TypeTensor) and isinstance(new_tp, TypeTensor):
            if len(old_tp.dims) != len(new_tp.dims):
                return
            for old_dim, new_dim in zip(old_tp.dims, new_tp.dims, strict=True):
                if isinstance(old_dim, str) and old_dim != new_dim:
                    dim_subst.setdefault(old_dim, new_dim)
            return
        if isinstance(old_tp, TypeNamed) and isinstance(new_tp, TypeNamed):
            if old_tp.name != new_tp.name or len(old_tp.args) != len(new_tp.args):
                return
            for old_dim, new_dim in zip(old_tp.args, new_tp.args, strict=True):
                if isinstance(old_dim, str) and old_dim != new_dim:
                    dim_subst.setdefault(old_dim, new_dim)

    params: list[AxonParam] = []
    value_arg_offset = len(module.path_params)
    for idx, param in enumerate(module.params):
        arg_idx = value_arg_offset + idx
        refined = callsite.arg_types[arg_idx] if arg_idx < len(callsite.arg_types) else None
        if refined is None:
            refined = callsite.kwarg_types.get(param.name)
        if param.optional and isinstance(refined, TypeOptional):
            refined = refined.inner
        preferred = _more_specific_type(param.type_expr, refined) or param.type_expr
        collect_dim_subst(param.type_expr, preferred)
        params.append(
            replace(
                param,
                type_expr=_clone_type_expr(preferred) if preferred is not None else None,
            )
        )
    preferred_return = _more_specific_type(module.return_type_expr, callsite.result_type)
    collect_dim_subst(module.return_type_expr, preferred_return)
    return replace(
        module,
        params=tuple(params),
        statements=tuple(
            _substitute_statement_dim_names(statement, subst=dim_subst)
            for statement in module.statements
        )
        if dim_subst
        else module.statements,
        return_type_expr=_return_type_from_callsite_result(preferred_return)
        or module.return_type_expr,
    )


def _specialize_generated_scc_external_entrypoints(program: AxonFile) -> tuple[AxonFile, bool]:
    graph = _module_call_graph(program)
    modules_by_name = {module.name: module for module in program.modules}
    replacements: dict[str, AxonDefinition] = {}
    for component_tuple in _module_sccs(graph):
        component = frozenset(component_tuple)
        is_recursive = len(component) > 1 or any(name in graph.get(name, ()) for name in component)
        if (
            not component
            or not is_recursive
            or not all(_is_loop_helper(name) for name in component)
        ):
            continue
        calls = _collect_external_component_callsites(program, component)
        if len(calls) != 1:
            continue
        callsite = calls[0]
        module = modules_by_name.get(callsite.callee)
        if module is None:
            continue
        for name in component:
            candidate = modules_by_name.get(name)
            if candidate is None:
                continue
            value_arg_offset = len(candidate.path_params)
            if len(candidate.params) + value_arg_offset != len(callsite.arg_types):
                continue
            specialized = _specialize_module_from_callsite(candidate, callsite)
            if not ast_equal(specialized, candidate):
                replacements[candidate.name] = specialized
    if not replacements:
        return program, False
    return (
        replace(
            program,
            modules=tuple(replacements.get(module.name, module) for module in program.modules),
        ),
        True,
    )


def _typecheck_flat_fixpoint(program: AxonFile) -> AxonFile:
    current = program
    max_iterations = 12
    for _ in range(max_iterations):
        typed = _typecheck_flat_once(current)
        specialized, specialized_changed = _specialize_generated_scc_external_entrypoints(typed)
        refined, refined_changed = _apply_global_generated_callsite_refinements(specialized)
        changed = specialized_changed or refined_changed
        next_program = refined if changed else typed
        if ast_equal(next_program, current):
            return next_program
        if not changed and ast_equal(next_program, typed):
            # Run at least one full AST equality check above. If global refinements
            # are inactive and the typed output differs from the input only because
            # inferred metadata was added, the next iteration must prove it stable.
            current = next_program
            continue
        current = next_program
    raise ValueError("Axon typecheck failed: typed AST fixpoint did not converge")


def _reject_bare_tensor_type(tp: TypeExpr | None, *, owner: str) -> None:
    if tp is None:
        return
    if isinstance(tp, TypeNamed) and tp.name == "Tensor" and not tp.args:
        raise ValueError(f"Axon typecheck failed in {owner}: Tensor type requires shape dims")
    if isinstance(tp, TypeOptional):
        _reject_bare_tensor_type(tp.inner, owner=owner)
        return
    if isinstance(tp, TypeList):
        _reject_bare_tensor_type(tp.item, owner=owner)
        return
    if isinstance(tp, TypeTuple):
        for item in tp.items:
            _reject_bare_tensor_type(item, owner=owner)


def _reject_bare_tensor_types(program: AxonFile) -> None:
    for name, alias in program.type_aliases.items():
        _reject_bare_tensor_type(alias.value, owner=f"type alias {name!r}")
    for module in program.modules:
        for param in module.params:
            _reject_bare_tensor_type(
                param.type_expr,
                owner=f"module {module.name!r} parameter {param.name!r}",
            )
        _reject_bare_tensor_type(module.return_type_expr, owner=f"module {module.name!r} return")


__all__: list[str] = []
