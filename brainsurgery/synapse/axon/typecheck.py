from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..ops import (
    get_op_lowering_known_output_arity,
    get_op_lowering_signature,
    get_op_lowering_type_signature,
)
from ..type_inference import infer_output_types_for_node
from .type_system import (
    TYPING_RULES,
    DimExprBinary,
    DimToken,
    TypeAny,
    TypeBool,
    TypeExpr,
    TypeFloat,
    TypeInt,
    TypeList,
    TypeNamed,
    TypeNull,
    TypeOptional,
    TypeString,
    TypeTensor,
    TypeTuple,
    dim_token_names,
    is_bool_like,
    is_numeric_type,
    parse_dim_expr,
    parse_type_expr,
    render_dim_token,
    render_type,
)
from .types import (
    AxonBind,
    AxonExpr,
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
    AxonExprPipe,
    AxonExprString,
    AxonExprTernary,
    AxonExprTuple,
    AxonKwargValue,
    AxonModule,
    AxonRepeat,
    AxonReturn,
    AxonScopeBind,
    AxonStatement,
)

_TYPE_EXPR_CLASSES = (
    TypeAny,
    TypeInt,
    TypeFloat,
    TypeBool,
    TypeNull,
    TypeString,
    TypeNamed,
    TypeOptional,
    TypeTensor,
    TypeList,
    TypeTuple,
)

_IMPLICIT_ACTIVATION_ALIASES: dict[str, str] = {
    "gelu": "_activations_gelu",
    "gelu_new": "_activations_gelu_new",
    "gelu_fast": "_activations_gelu_new",
    "gelu_pytorch_tanh": "_activations_gelu_pytorch_tanh",
    "gegelu": "_activations_gegelu",
    "relu": "_activations_relu",
    "sigmoid": "_activations_sigmoid",
    "silu": "_activations_silu",
    "swiglu": "_activations_swiglu",
}

_NAMED_LIST_ITEM_TYPES: dict[str, TypeExpr] = {
    "List": TypeAny(),
}
_LIST_NAMED_ALIASES: set[str] = set(_NAMED_LIST_ITEM_TYPES)


def _named_list_item_type(
    name: str, *, type_aliases: dict[str, TypeExpr] | None = None
) -> TypeExpr | None:
    alias_map = type_aliases if isinstance(type_aliases, dict) else {}
    builtin = _NAMED_LIST_ITEM_TYPES.get(name)
    if builtin is not None:
        return builtin
    alias = alias_map.get(name)
    if alias is None:
        return None
    expanded = _resolve_type_aliases(alias, type_aliases=alias_map)
    if isinstance(expanded, TypeList):
        return expanded.item
    return None


def _build_type_alias_table(modules: tuple[AxonModule, ...]) -> dict[str, TypeExpr]:
    table: dict[str, TypeExpr] = {}
    for module in modules:
        alias_map = module.type_aliases if isinstance(module.type_aliases, dict) else {}
        for name, alias in alias_map.items():
            if not isinstance(name, str) or not isinstance(alias, _TYPE_EXPR_CLASSES):
                continue
            prev = table.get(name)
            if prev is None:
                table[name] = alias
                continue
            if prev != alias:
                raise ValueError(
                    f"Axon typecheck failed: conflicting type alias {name!r} across modules"
                )
    return table


def _resolve_type_aliases(
    tp: TypeExpr,
    *,
    type_aliases: dict[str, TypeExpr],
    stack: tuple[str, ...] = (),
) -> TypeExpr:
    if isinstance(tp, TypeNamed):
        alias = type_aliases.get(tp.name)
        if alias is None:
            return tp
        if tp.name in stack:
            cycle = " -> ".join((*stack, tp.name))
            raise ValueError(f"Axon typecheck failed: cyclic type alias detected: {cycle}")
        return _resolve_type_aliases(alias, type_aliases=type_aliases, stack=(*stack, tp.name))
    if isinstance(tp, TypeOptional):
        return TypeOptional(
            inner=_resolve_type_aliases(tp.inner, type_aliases=type_aliases, stack=stack)
        )
    if isinstance(tp, TypeList):
        return TypeList(item=_resolve_type_aliases(tp.item, type_aliases=type_aliases, stack=stack))
    if isinstance(tp, TypeTuple):
        return TypeTuple(
            items=tuple(
                _resolve_type_aliases(item, type_aliases=type_aliases, stack=stack)
                for item in tp.items
            )
        )
    return tp


@dataclass(frozen=True)
class ModuleSignature:
    name: str
    path_param_count: int
    params: tuple[TypeExpr, ...]
    param_names: tuple[str, ...]
    returns: tuple[TypeExpr, ...]
    optional_params: tuple[bool, ...]
    param_default_exprs: tuple[AxonExpr | None, ...]
    param_shapes: tuple[tuple[DimToken, ...] | None, ...]
    return_shapes: tuple[tuple[DimToken, ...] | None, ...]


def _stmt_path(path: tuple[int, ...]) -> str:
    if not path:
        return "root"
    return "root." + ".".join(str(i) for i in path)


def _error(module: AxonModule, path: tuple[int, ...], message: str) -> ValueError:
    return ValueError(
        f"Axon typecheck failed in module '{module.name}' at {_stmt_path(path)}: {message}"
    )


def _shape_dims_from_type(tp: TypeExpr) -> tuple[DimToken, ...] | None:
    if isinstance(tp, TypeOptional):
        return _shape_dims_from_type(tp.inner)
    if isinstance(tp, TypeTensor):
        return tp.dims
    return None


def _shape_from_param_shape(shape: tuple[DimToken, ...] | None) -> tuple[DimToken, ...] | None:
    if shape is None:
        return None
    return tuple(shape)


def _module_signature(module: AxonModule, *, type_aliases: dict[str, TypeExpr]) -> ModuleSignature:
    param_types: list[TypeExpr] = []
    param_names: list[str] = []
    optional_flags: list[bool] = []
    default_exprs: list[AxonExpr | None] = []
    param_shapes: list[tuple[DimToken, ...] | None] = []
    for param in module.params:
        if not isinstance(param.type_expr, _TYPE_EXPR_CLASSES):
            raise ValueError(
                f"Axon typecheck failed in module '{module.name}': parameter {param.name!r} is missing a declared type"
            )
        param_type = _resolve_type_aliases(param.type_expr, type_aliases=type_aliases)
        if param.optional:
            param_type = TypeOptional(param_type)
        param_types.append(param_type)
        param_names.append(param.name)
        optional_flags.append(bool(param.optional))
        default_exprs.append(param.default_expr)
        parsed_shape = _shape_from_param_shape(param.shape)
        if parsed_shape is None:
            parsed_shape = _shape_dims_from_type(param_type)
        param_shapes.append(parsed_shape)

    return_decl = module.return_type_expr
    if not isinstance(return_decl, _TYPE_EXPR_CLASSES):
        raise ValueError(
            f"Axon typecheck failed in module '{module.name}': missing declared return type"
        )
    resolved_return = _resolve_type_aliases(return_decl, type_aliases=type_aliases)
    return_types = (
        list(resolved_return.items) if isinstance(resolved_return, TypeTuple) else [resolved_return]
    )
    return_shapes = [_shape_dims_from_type(tp) for tp in return_types]
    return ModuleSignature(
        name=module.name,
        path_param_count=len(module.path_params),
        params=tuple(param_types),
        param_names=tuple(param_names),
        returns=tuple(return_types),
        optional_params=tuple(optional_flags),
        param_default_exprs=tuple(default_exprs),
        param_shapes=tuple(param_shapes),
        return_shapes=tuple(return_shapes),
    )


def _extract_primitive_aliases(
    modules: tuple[AxonModule, ...], *, type_aliases: dict[str, TypeExpr]
) -> dict[str, tuple[str, int]]:
    def _module_path_param_names(module: AxonModule) -> tuple[str, ...]:
        if module.path_params:
            return tuple(module.path_params)
        if module.path_param is not None:
            return (module.path_param,)
        return ()

    def _is_identity_alias_call(module: AxonModule, value: AxonExprCall) -> bool:
        # Defaulted/optional wrapper params change call-surface semantics
        # (especially kwargs), so do not collapse those wrappers to primitives.
        if any(param.optional for param in module.params):
            return False
        if any(param.default_expr is not None for param in module.params):
            return False
        callee_parts = value.callee.split("@")
        callee_path_params = tuple(callee_parts[1:])
        module_path_params = _module_path_param_names(module)
        if callee_path_params != module_path_params:
            return False
        if value.kwargs:
            return False
        if len(value.args) != len(module.params):
            return False
        for arg_expr, param in zip(value.args, module.params, strict=True):
            if not isinstance(arg_expr, AxonExprName) or arg_expr.name != param.name:
                return False
        return True

    def _same_module_signature(left: AxonModule, right: AxonModule) -> bool:
        left_sig = _module_signature(left, type_aliases=type_aliases)
        right_sig = _module_signature(right, type_aliases=type_aliases)
        return (
            left_sig.path_param_count == right_sig.path_param_count
            and left_sig.params == right_sig.params
            and left_sig.param_names == right_sig.param_names
            and left_sig.returns == right_sig.returns
            and left_sig.optional_params == right_sig.optional_params
            and left_sig.param_default_exprs == right_sig.param_default_exprs
            and left_sig.param_shapes == right_sig.param_shapes
            and left_sig.return_shapes == right_sig.return_shapes
        )

    modules_by_name = {module.name: module for module in modules}
    called_with_kwargs: set[str] = set()
    for module in modules:
        for stmt in module.statements:
            values: tuple[AxonExpr, ...] = ()
            if isinstance(stmt, AxonBind):
                values = (stmt.expr,)
            elif isinstance(stmt, AxonReturn):
                values = tuple(stmt.values)
            if not values:
                continue
            stack: list[AxonExpr] = list(values)
            while stack:
                expr = stack.pop()
                if isinstance(expr, AxonExprCall):
                    if expr.kwargs:
                        called_with_kwargs.add(expr.callee.split("@", 1)[0])
                    stack.extend(expr.args)
                    for kw_value in expr.kwargs.values():
                        if isinstance(kw_value, AxonExpr):
                            stack.append(kw_value)
                elif isinstance(expr, AxonExprParen):
                    stack.append(expr.inner)
                elif isinstance(expr, AxonExprPipe):
                    stack.append(expr.value)
                    stack.extend(expr.stages)
                elif isinstance(expr, AxonExprBinary):
                    stack.append(expr.left)
                    stack.append(expr.right)
                elif isinstance(expr, AxonExprIf):
                    stack.append(expr.cond)
                    stack.append(expr.true_expr)
                    stack.append(expr.false_expr)
                elif isinstance(expr, AxonExprTernary):
                    stack.append(expr.cond)
                    stack.append(expr.true_expr)
                    stack.append(expr.false_expr)
                elif isinstance(expr, AxonExprList):
                    stack.extend(expr.items)
                elif isinstance(expr, AxonExprTuple):
                    stack.extend(expr.items)
                elif isinstance(expr, AxonExprDo):
                    for do_stmt in expr.body:
                        if isinstance(do_stmt, AxonBind):
                            stack.append(do_stmt.expr)
                        elif isinstance(do_stmt, AxonReturn):
                            stack.extend(do_stmt.values)
                elif isinstance(expr, AxonExprLambda):
                    stack.append(expr.body)
    direct_aliases: dict[str, tuple[str, int]] = {}
    for module in modules:
        if not isinstance(module.name, str) or "." not in module.name:
            continue
        if len(module.statements) != 1:
            continue
        stmt = module.statements[0]
        if not isinstance(stmt, AxonReturn) or len(stmt.values) != 1:
            continue
        value = stmt.values[0]
        if not isinstance(value, AxonExprCall):
            continue
        if not _is_identity_alias_call(module, value):
            continue
        module_base_name = module.name.rsplit(".", 1)[-1]
        if module.name in called_with_kwargs or module_base_name in called_with_kwargs:
            continue
        target_base = value.callee.split("@", 1)[0]
        target_module = modules_by_name.get(target_base)
        if target_module is not None and not _same_module_signature(module, target_module):
            continue
        direct_aliases[module.name] = (target_base, len(_module_path_param_names(module)))

    aliases: dict[str, tuple[str, int]] = {}
    for name, (target_base, expected_path_count) in direct_aliases.items():
        seen: set[str] = set()
        resolved = target_base
        while not resolved.startswith("_"):
            if resolved in seen:
                break
            seen.add(resolved)
            next_alias = direct_aliases.get(resolved)
            if next_alias is None:
                break
            next_base, next_path_count = next_alias
            if next_path_count != expected_path_count:
                break
            resolved = next_base
        if resolved.startswith("_"):
            aliases[name] = (resolved, expected_path_count)
    return aliases


def _symbol_dim_table(
    module: AxonModule, *, type_aliases: dict[str, TypeExpr]
) -> dict[str, DimToken]:
    signature_dim_names = _module_signature_dim_names(module, type_aliases=type_aliases)
    table: dict[str, DimToken] = {}
    if not isinstance(module.symbols, dict):
        return table
    for name, value in module.symbols.items():
        if not isinstance(name, str):
            continue
        if name in signature_dim_names:
            # Signature dimensions are polymorphic variables; do not force them
            # to global symbol literals during unification.
            continue
        if isinstance(value, int):
            table[name] = value
            continue
        if isinstance(value, float) and value.is_integer():
            table[name] = int(value)
            continue
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                continue
            try:
                table[name] = parse_dim_expr(raw)
            except ValueError:
                continue
    return table


def _module_signature_dim_names(
    module: AxonModule, *, type_aliases: dict[str, TypeExpr]
) -> set[str]:
    def _type_dim_names(tp: TypeExpr) -> set[str]:
        if isinstance(tp, TypeOptional):
            return _type_dim_names(tp.inner)
        if isinstance(tp, TypeTensor):
            names: set[str] = set()
            for dim in tp.dims:
                names.update(dim_token_names(dim))
            return names
        if isinstance(tp, TypeList):
            return _type_dim_names(tp.item)
        if isinstance(tp, TypeTuple):
            tuple_names: set[str] = set()
            for item in tp.items:
                tuple_names.update(_type_dim_names(item))
            return tuple_names
        return set()

    signature_dim_names: set[str] = set()
    for param in module.params:
        if isinstance(param.type_expr, _TYPE_EXPR_CLASSES):
            resolved_param_type = _resolve_type_aliases(param.type_expr, type_aliases=type_aliases)
            signature_dim_names.update(_type_dim_names(resolved_param_type))
        if isinstance(param.shape, tuple):
            for dim in param.shape:
                signature_dim_names.update(dim_token_names(dim))
    if isinstance(module.return_type_expr, _TYPE_EXPR_CLASSES):
        return_decl = _resolve_type_aliases(module.return_type_expr, type_aliases=type_aliases)
        if isinstance(return_decl, TypeTuple):
            for ret in return_decl.items:
                signature_dim_names.update(_type_dim_names(ret))
        else:
            signature_dim_names.update(_type_dim_names(return_decl))
    if isinstance(module.return_shape, tuple):
        for dim in module.return_shape:
            signature_dim_names.update(dim_token_names(dim))
    return signature_dim_names


def _flatten_dim_op(token: DimToken, op: str) -> list[DimToken]:
    if isinstance(token, DimExprBinary) and token.op == op:
        return [*_flatten_dim_op(token.left, op), *_flatten_dim_op(token.right, op)]
    return [token]


def _normalize_dim(
    token: DimToken,
    *,
    subst: dict[str, DimToken],
    symbols: dict[str, DimToken],
    visiting: set[str] | None = None,
    visiting_expr: set[int] | None = None,
) -> DimToken:
    if isinstance(token, int):
        return token
    if isinstance(token, str):
        seen = visiting or set()
        bound = subst.get(token)
        if bound is not None:
            if token in seen:
                return token
            return _normalize_dim(
                bound,
                subst=subst,
                symbols=symbols,
                visiting={*seen, token},
                visiting_expr=visiting_expr,
            )
        if token in symbols and token not in seen:
            expanded = symbols[token]
            return _normalize_dim(
                expanded,
                subst=subst,
                symbols=symbols,
                visiting={*seen, token},
                visiting_expr=visiting_expr,
            )
        return token
    assert isinstance(token, DimExprBinary)
    seen_expr = visiting_expr or set()
    token_id = id(token)
    if token_id in seen_expr:
        return token
    next_seen_expr = {*seen_expr, token_id}
    left = _normalize_dim(
        token.left,
        subst=subst,
        symbols=symbols,
        visiting=visiting,
        visiting_expr=next_seen_expr,
    )
    right = _normalize_dim(
        token.right,
        subst=subst,
        symbols=symbols,
        visiting=visiting,
        visiting_expr=next_seen_expr,
    )
    if token.op in {"+", "*"}:
        terms: list[DimToken] = []
        for part in _flatten_dim_op(DimExprBinary(op=token.op, left=left, right=right), token.op):
            terms.append(part)
        int_terms = [term for term in terms if isinstance(term, int)]
        non_int_terms: list[DimToken] = [term for term in terms if not isinstance(term, int)]
        if token.op == "+":
            int_sum = sum(int_terms)
            if int_sum != 0 or not non_int_terms:
                non_int_terms.append(int_sum)
        else:
            int_prod = 1
            for part in int_terms:
                int_prod *= part
            if int_prod == 0:
                return 0
            if int_prod != 1 or not non_int_terms:
                non_int_terms.append(int_prod)
        non_int_terms.sort(key=render_dim_token)
        out = non_int_terms[0]
        for part in non_int_terms[1:]:
            out = DimExprBinary(op=token.op, left=out, right=part)
        return out
    if isinstance(left, int) and isinstance(right, int):
        if token.op == "-":
            return left - right
        if token.op == "/":
            if right != 0 and left % right == 0:
                return left // right
            return DimExprBinary(op=token.op, left=left, right=right)
    return DimExprBinary(op=token.op, left=left, right=right)


def _dim_key(token: DimToken) -> tuple[Any, ...]:
    if isinstance(token, int):
        return ("int", token)
    if isinstance(token, str):
        return ("name", token)
    assert isinstance(token, DimExprBinary)
    return ("op", token.op, _dim_key(token.left), _dim_key(token.right))


def _dim_equal(
    left: DimToken,
    right: DimToken,
    *,
    subst: dict[str, DimToken],
    symbols: dict[str, DimToken],
) -> bool:
    left_n = _normalize_dim(left, subst=subst, symbols=symbols)
    right_n = _normalize_dim(right, subst=subst, symbols=symbols)
    return _dim_key(left_n) == _dim_key(right_n)


def _substitute_type_dims(
    tp: TypeExpr,
    *,
    dim_subst: dict[str, DimToken],
    symbols: dict[str, DimToken],
    rest_subst: dict[str, tuple[DimToken, ...]] | None = None,
) -> TypeExpr:
    rest_map = rest_subst if isinstance(rest_subst, dict) else {}
    if isinstance(tp, TypeOptional):
        return TypeOptional(
            inner=_substitute_type_dims(
                tp.inner,
                dim_subst=dim_subst,
                symbols=symbols,
                rest_subst=rest_map,
            )
        )
    if isinstance(tp, TypeTensor):
        dims_out: list[DimToken] = []
        has_unresolved_rest = False
        for dim in tp.dims:
            if isinstance(dim, str) and dim.startswith(".."):
                bound = rest_map.get(dim)
                if isinstance(bound, tuple):
                    dims_out.extend(
                        _normalize_dim(tok, subst=dim_subst, symbols=symbols) for tok in bound
                    )
                else:
                    # If a variadic rest dim is unbound (for example caller
                    # provided only generic Tensor with unknown rank), degrade
                    # to rank-unknown Tensor/IdxTensor.
                    has_unresolved_rest = True
                continue
            dims_out.append(_normalize_dim(dim, subst=dim_subst, symbols=symbols))
        if has_unresolved_rest:
            return TypeNamed(name=tp.base)
        dims = tuple(dims_out)
        return TypeTensor(base=tp.base, dims=dims)
    if isinstance(tp, TypeList):
        return TypeList(
            item=_substitute_type_dims(
                tp.item,
                dim_subst=dim_subst,
                symbols=symbols,
                rest_subst=rest_map,
            )
        )
    if isinstance(tp, TypeTuple):
        return TypeTuple(
            items=tuple(
                _substitute_type_dims(
                    item,
                    dim_subst=dim_subst,
                    symbols=symbols,
                    rest_subst=rest_map,
                )
                for item in tp.items
            )
        )
    return tp


def _is_flexible_dim_var(token: DimToken, rigid_symbols: set[str]) -> bool:
    return isinstance(token, str) and token not in rigid_symbols


def _is_rest_dim_var(token: DimToken) -> bool:
    return isinstance(token, str) and token.startswith("..") and len(token) > 2


def _dims_compatible(
    actual: tuple[DimToken, ...],
    expected: tuple[DimToken, ...],
    subst: dict[str, DimToken],
    *,
    symbols: dict[str, DimToken],
    rigid_symbols: set[str],
    rest_subst: dict[str, tuple[DimToken, ...]] | None = None,
) -> bool:
    rest_map = rest_subst if isinstance(rest_subst, dict) else {}
    expected_rest = [i for i, tok in enumerate(expected) if _is_rest_dim_var(tok)]
    if len(expected_rest) > 1:
        return False
    if expected_rest:
        rest_idx = expected_rest[0]
        rest_tok = expected[rest_idx]
        assert isinstance(rest_tok, str)
        prefix = expected[:rest_idx]
        suffix = expected[rest_idx + 1 :]
        if len(actual) < len(prefix) + len(suffix):
            return False
        if prefix and not _dims_compatible(
            actual[: len(prefix)],
            prefix,
            subst,
            symbols=symbols,
            rigid_symbols=rigid_symbols,
            rest_subst=rest_map,
        ):
            return False
        if suffix and not _dims_compatible(
            actual[len(actual) - len(suffix) :],
            suffix,
            subst,
            symbols=symbols,
            rigid_symbols=rigid_symbols,
            rest_subst=rest_map,
        ):
            return False
        middle = actual[len(prefix) : len(actual) - len(suffix)]
        prev_rest = rest_map.get(rest_tok)
        if prev_rest is None:
            rest_map[rest_tok] = middle
            return True
        if len(prev_rest) != len(middle):
            return False
        for left, right in zip(prev_rest, middle, strict=True):
            if not _dim_equal(left, right, subst=subst, symbols=symbols):
                return False
        return True
    if len(actual) != len(expected):
        return False
    for left_raw, right_raw in zip(actual, expected, strict=True):
        left = _normalize_dim(left_raw, subst=subst, symbols=symbols)
        right = _normalize_dim(right_raw, subst=subst, symbols=symbols)
        if _is_flexible_dim_var(right, rigid_symbols):
            assert isinstance(right, str)
            prev_dim: DimToken | None = subst.get(right)
            if prev_dim is None:
                if isinstance(left, str) and left == right:
                    continue
                subst[right] = left
                continue
            if not _dim_equal(prev_dim, left, subst=subst, symbols=symbols):
                return False
            continue
        if _is_flexible_dim_var(left, rigid_symbols):
            assert isinstance(left, str)
            prev_dim_left: DimToken | None = subst.get(left)
            if prev_dim_left is None:
                if isinstance(right, str) and right == left:
                    continue
                subst[left] = right
                continue
            if not _dim_equal(prev_dim_left, right, subst=subst, symbols=symbols):
                return False
            continue
        if not _dim_equal(left, right, subst=subst, symbols=symbols):
            return False
    return True


def _types_compatible(
    actual: TypeExpr,
    expected: TypeExpr,
    *,
    dim_subst: dict[str, DimToken],
    dim_symbols: dict[str, DimToken],
    rigid_symbols: set[str],
    rest_subst: dict[str, tuple[DimToken, ...]] | None = None,
) -> bool:
    rest_map = rest_subst if isinstance(rest_subst, dict) else {}

    def _is_path_type(tp: TypeExpr) -> bool:
        return isinstance(tp, TypeNamed) and tp.name == "Path"

    # Path-typed parameters accept explicit string/path values only.
    if _is_path_type(expected):
        if isinstance(actual, TypeString):
            return True
        if _is_path_type(actual):
            return True
    if _is_path_type(actual) and isinstance(expected, TypeString):
        return True

    if isinstance(expected, TypeAny):
        return True
    if isinstance(actual, TypeAny):
        return False
    if isinstance(expected, TypeOptional):
        if isinstance(actual, TypeOptional):
            return _types_compatible(
                actual.inner,
                expected.inner,
                dim_subst=dim_subst,
                dim_symbols=dim_symbols,
                rigid_symbols=rigid_symbols,
                rest_subst=rest_map,
            )
        if isinstance(actual, TypeNull):
            return True
        return _types_compatible(
            actual,
            expected.inner,
            dim_subst=dim_subst,
            dim_symbols=dim_symbols,
            rigid_symbols=rigid_symbols,
            rest_subst=rest_map,
        )
    if isinstance(actual, TypeOptional):
        if isinstance(expected, TypeOptional):
            return _types_compatible(
                actual.inner,
                expected.inner,
                dim_subst=dim_subst,
                dim_symbols=dim_symbols,
                rigid_symbols=rigid_symbols,
                rest_subst=rest_map,
            )
        return False
    if isinstance(actual, TypeNull):
        if isinstance(expected, TypeNull | TypeOptional | TypeAny):
            return True
        return False
    if isinstance(actual, TypeTensor) and isinstance(expected, TypeTensor):
        if actual.base != expected.base:
            if not {actual.base, expected.base} <= {"Tensor", "IdxTensor"}:
                return False
        return _dims_compatible(
            actual.dims,
            expected.dims,
            dim_subst,
            symbols=dim_symbols,
            rigid_symbols=rigid_symbols,
            rest_subst=rest_map,
        )
    if isinstance(actual, TypeTuple) and isinstance(expected, TypeTuple):
        if len(actual.items) != len(expected.items):
            return False
        return all(
            _types_compatible(
                a,
                e,
                dim_subst=dim_subst,
                dim_symbols=dim_symbols,
                rigid_symbols=rigid_symbols,
                rest_subst=rest_map,
            )
            for a, e in zip(actual.items, expected.items, strict=True)
        )
    if isinstance(actual, TypeList) and isinstance(expected, TypeList):
        return _types_compatible(
            actual.item,
            expected.item,
            dim_subst=dim_subst,
            dim_symbols=dim_symbols,
            rigid_symbols=rigid_symbols,
            rest_subst=rest_map,
        )
    if isinstance(actual, TypeInt) and isinstance(expected, TypeFloat):
        return True
    if isinstance(actual, TypeInt) and isinstance(expected, TypeNamed) and expected.name == "Dim":
        return True
    if isinstance(actual, TypeList) and isinstance(expected, TypeNamed):
        expected_item = _named_list_item_type(expected.name)
        if expected_item is None:
            return False
        if isinstance(actual.item, TypeAny):
            return True
        if isinstance(expected_item, TypeAny):
            return True
        return _types_compatible(
            actual.item,
            expected_item,
            dim_subst=dim_subst,
            dim_symbols=dim_symbols,
            rigid_symbols=rigid_symbols,
            rest_subst=rest_map,
        )
    if isinstance(actual, TypeNamed) and isinstance(expected, TypeList):
        actual_item = _named_list_item_type(actual.name)
        if actual_item is None:
            return False
        if isinstance(actual_item, TypeAny):
            return True
        return _types_compatible(
            actual_item,
            expected.item,
            dim_subst=dim_subst,
            dim_symbols=dim_symbols,
            rigid_symbols=rigid_symbols,
            rest_subst=rest_map,
        )
    if isinstance(actual, TypeTensor) and isinstance(expected, TypeNamed):
        return expected.name in {"Tensor", "IdxTensor"}
    if isinstance(actual, TypeNamed) and isinstance(expected, TypeTensor):
        return actual.name in {"Tensor", "IdxTensor"}
    if isinstance(actual, TypeNamed) and _is_type_var_name(actual.name):
        return True
    if type(actual) is type(expected):
        if isinstance(actual, TypeNamed) and isinstance(expected, TypeNamed):
            if {actual.name, expected.name} <= {"Tensor", "IdxTensor"}:
                return True
            return actual.name == expected.name
        return True
    if isinstance(actual, TypeNamed) or isinstance(expected, TypeNamed):
        return False
    return False


def _is_type_var_name(name: str) -> bool:
    # Preferred convention: leading underscore + all-caps (e.g. _T, _ELEM).
    # Keep legacy support for bare all-caps names (e.g. T, U).
    # Nominal types (Tensor, Cache, Path, CacheLayer, ...) are not all-caps.
    if not name:
        return False
    if name.startswith("_"):
        tail = name[1:]
        return bool(tail) and tail.isupper()
    return name.isupper()


def _substitute_type_vars(tp: TypeExpr, type_subst: dict[str, TypeExpr]) -> TypeExpr:
    if isinstance(tp, TypeNamed):
        bound = type_subst.get(tp.name)
        return bound if bound is not None else tp
    if isinstance(tp, TypeOptional):
        inner = _substitute_type_vars(tp.inner, type_subst)
        if isinstance(inner, TypeOptional):
            return inner
        return TypeOptional(inner=inner)
    if isinstance(tp, TypeList):
        return TypeList(item=_substitute_type_vars(tp.item, type_subst))
    if isinstance(tp, TypeTuple):
        return TypeTuple(items=tuple(_substitute_type_vars(item, type_subst) for item in tp.items))
    if isinstance(tp, TypeTensor):
        return tp
    return tp


def _bind_type_vars(
    actual: TypeExpr,
    expected: TypeExpr,
    *,
    type_subst: dict[str, TypeExpr],
    dim_subst: dict[str, DimToken],
    dim_symbols: dict[str, DimToken],
    rigid_symbols: set[str],
    rest_subst: dict[str, tuple[DimToken, ...]] | None = None,
) -> bool:
    rest_map = rest_subst if rest_subst is not None else {}
    if isinstance(expected, TypeNamed) and _is_type_var_name(expected.name):
        bound = type_subst.get(expected.name)
        if bound is None:
            type_subst[expected.name] = actual
            return True
        return _types_compatible(
            actual,
            bound,
            dim_subst=dim_subst,
            dim_symbols=dim_symbols,
            rigid_symbols=rigid_symbols,
            rest_subst=rest_map,
        )
    if isinstance(expected, TypeOptional):
        if isinstance(actual, TypeNull):
            return True
        if isinstance(actual, TypeOptional):
            return _bind_type_vars(
                actual.inner,
                expected.inner,
                type_subst=type_subst,
                dim_subst=dim_subst,
                dim_symbols=dim_symbols,
                rigid_symbols=rigid_symbols,
                rest_subst=rest_map,
            )
        return _bind_type_vars(
            actual,
            expected.inner,
            type_subst=type_subst,
            dim_subst=dim_subst,
            dim_symbols=dim_symbols,
            rigid_symbols=rigid_symbols,
            rest_subst=rest_map,
        )
    if isinstance(actual, TypeOptional):
        return False
    if isinstance(expected, TypeList):
        actual_list = actual
        if isinstance(actual_list, TypeNamed):
            alias_item = _named_list_item_type(actual_list.name)
            if alias_item is None:
                return False
            actual_list = TypeList(item=alias_item)
        if not isinstance(actual_list, TypeList):
            return False
        return _bind_type_vars(
            actual_list.item,
            expected.item,
            type_subst=type_subst,
            dim_subst=dim_subst,
            dim_symbols=dim_symbols,
            rigid_symbols=rigid_symbols,
            rest_subst=rest_map,
        )
    if isinstance(actual, TypeNamed) and isinstance(expected, TypeTensor):
        return actual.name in {"Tensor", "IdxTensor"}
    if isinstance(actual, TypeTensor) and isinstance(expected, TypeNamed):
        if _is_type_var_name(expected.name):
            bound = type_subst.get(expected.name)
            if bound is None:
                type_subst[expected.name] = actual
                return True
            return _types_compatible(
                actual,
                bound,
                dim_subst=dim_subst,
                dim_symbols=dim_symbols,
                rigid_symbols=rigid_symbols,
                rest_subst=rest_map,
            )
        return expected.name in {"Tensor", "IdxTensor"}
    return _types_compatible(
        actual,
        _substitute_type_vars(expected, type_subst),
        dim_subst=dim_subst,
        dim_symbols=dim_symbols,
        rigid_symbols=rigid_symbols,
        rest_subst=rest_map,
    )


def _is_int_value(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _kwarg_matches_kind(value: Any, expected: str) -> bool:
    expr_type: TypeExpr | None = None
    expr_dim: DimToken | None = None
    raw = value
    if (
        isinstance(value, tuple)
        and len(value) == 3
        and value[0] == "__expr__"
        and isinstance(value[1], _TYPE_EXPR_CLASSES)
    ):
        expr_type = value[1]
        if isinstance(value[2], int | str | DimExprBinary):
            expr_dim = value[2]
        raw = None

    if expected == "bool":
        if isinstance(expr_type, TypeBool):
            return True
        return isinstance(raw, bool)
    if expected == "int":
        if isinstance(expr_type, TypeInt):
            return True
        return _is_int_value(raw)
    if expected == "number":
        if isinstance(expr_type, TypeInt | TypeFloat):
            return True
        if expr_dim is not None:
            return True
        return isinstance(raw, (int, float, str)) and not isinstance(raw, bool)
    if expected == "str":
        if isinstance(expr_type, TypeString):
            return True
        return isinstance(raw, str)
    if expected == "path":
        if isinstance(expr_type, TypeNamed) and expr_type.name == "Path":
            return True
        if isinstance(expr_type, TypeString):
            return True
        return isinstance(raw, str)
    if expected == "path_or_null":
        if isinstance(expr_type, TypeNamed) and expr_type.name == "Path":
            return True
        if isinstance(expr_type, (TypeString, TypeNull)):
            return True
        return isinstance(raw, str) or raw is None
    if expected == "str_or_bool_or_null":
        if isinstance(expr_type, (TypeString, TypeBool, TypeNull)):
            return True
        return isinstance(raw, (str, bool)) or raw is None
    if expected == "dim":
        if expr_dim is not None:
            return True
        if isinstance(expr_type, TypeInt | TypeFloat):
            return True
        return _is_int_value(raw) or isinstance(raw, str)
    if expected == "list_int":
        if isinstance(expr_type, TypeList):
            item = (
                expr_type.item.inner if isinstance(expr_type.item, TypeOptional) else expr_type.item
            )
            return isinstance(item, TypeInt)
        return isinstance(raw, list) and all(_is_int_value(v) for v in raw)
    if expected == "list_dim":
        if isinstance(expr_type, TypeList):
            item = (
                expr_type.item.inner if isinstance(expr_type.item, TypeOptional) else expr_type.item
            )
            return isinstance(item, TypeInt | TypeFloat | TypeNamed | TypeAny)
        return isinstance(raw, list) and all(_is_int_value(v) or isinstance(v, str) for v in raw)
    return True


def _is_any_type(tp: TypeExpr) -> bool:
    if isinstance(tp, TypeAny):
        return True
    if isinstance(tp, TypeNamed):
        return tp.name == "Any"
    if isinstance(tp, TypeOptional):
        return _is_any_type(tp.inner)
    if isinstance(tp, TypeList):
        return _is_any_type(tp.item)
    if isinstance(tp, TypeTuple):
        return bool(tp.items) and all(_is_any_type(item) for item in tp.items)
    return False


def _refine_env_for_null_test(
    cond: AxonExpr,
    env: dict[str, TypeExpr],
) -> tuple[dict[str, TypeExpr], dict[str, TypeExpr]]:
    true_env = dict(env)
    false_env = dict(env)
    while isinstance(cond, AxonExprParen):
        cond = cond.inner
    if not isinstance(cond, AxonExprBinary) or cond.op not in {"==", "!="}:
        return true_env, false_env

    refined_name: str | None = None
    refined_type: TypeExpr | None = None
    if isinstance(cond.left, AxonExprName) and isinstance(cond.right, AxonExprNull):
        refined_name = cond.left.name
        refined_type = env.get(refined_name)
    elif isinstance(cond.right, AxonExprName) and isinstance(cond.left, AxonExprNull):
        refined_name = cond.right.name
        refined_type = env.get(refined_name)

    if refined_name is None or not isinstance(refined_type, TypeOptional):
        return true_env, false_env

    if cond.op == "==":
        true_env[refined_name] = TypeNull()
        false_env[refined_name] = refined_type.inner
    else:
        true_env[refined_name] = refined_type.inner
        false_env[refined_name] = TypeNull()
    return true_env, false_env


def _first_tensor_like_type(arg_types: list[TypeExpr]) -> TypeExpr | None:
    for tp in arg_types:
        root = tp.inner if isinstance(tp, TypeOptional) else tp
        if isinstance(root, TypeTensor):
            return root
        if isinstance(root, TypeNamed) and root.name in {"Tensor", "IdxTensor"}:
            return root
    return None


def _primitive_fallback_type(
    op_name: str,
    *,
    arg_types: list[TypeExpr],
    arity: int,
) -> TypeExpr | None:
    tensor_like = _first_tensor_like_type(arg_types)
    tensor_default = tensor_like if tensor_like is not None else TypeNamed(name="Tensor")

    if arity == 1:
        if op_name == "cache_seq_len":
            return TypeInt()
        if op_name == "position_ids":
            return TypeNamed(name="IdxTensor")
        if op_name == "list_init":
            return TypeList(item=TypeAny())
        if op_name == "list_append":
            if arg_types:
                root = arg_types[0]
                if isinstance(root, TypeNamed):
                    alias_item = _named_list_item_type(root.name)
                    if alias_item is not None:
                        root = TypeList(item=alias_item)
                if isinstance(root, TypeList):
                    return root
            return TypeList(item=TypeAny())
        if op_name == "list_index":
            if arg_types:
                root = arg_types[0]
                if isinstance(root, TypeNamed):
                    alias_item = _named_list_item_type(root.name)
                    if alias_item is not None:
                        root = TypeList(item=alias_item)
                if isinstance(root, TypeList):
                    return root.item
        if op_name in {
            "activation",
            "add",
            "mul",
            "div",
            "clamp",
            "layernorm",
            "rmsnorm",
            "softmax",
            "zeros_like",
            "reshape",
            "unsqueeze",
            "merge_heads",
            "repeat",
            "reshape_heads",
            "attention",
            "causal_mask",
            "concat",
            "embedding",
            "linear",
            "linear_position_bias",
            "moe_grouped_ffn",
            "moe_scatter_add",
        }:
            return tensor_default
        return None

    if op_name == "cache_update" and arity == 3:
        present = _cache_update_present_type(arg_types[0]) if arg_types else tensor_default
        k_ctx = arg_types[1] if len(arg_types) > 1 else tensor_default
        v_ctx = arg_types[2] if len(arg_types) > 2 else tensor_default
        return TypeTuple(items=(k_ctx, v_ctx, present))
    if op_name == "moe_select" and arity == 4:
        return TypeTuple(
            items=(
                tensor_default,
                TypeNamed(name="IdxTensor"),
                TypeNamed(name="IdxTensor"),
                tensor_default,
            )
        )
    if op_name == "split_qkv_heads" and arity == 3:
        return TypeTuple(items=(tensor_default, tensor_default, tensor_default))
    if op_name == "rope_pair" and arity == 2:
        q = arg_types[0] if arg_types else tensor_default
        k = arg_types[1] if len(arg_types) > 1 else tensor_default
        return TypeTuple(items=(q, k))
    return None


def _cache_update_present_type(past_type: TypeExpr) -> TypeExpr:
    root = past_type.inner if isinstance(past_type, TypeOptional) else past_type
    if isinstance(root, TypeNamed) and root.name == "Cache":
        return TypeOptional(inner=TypeNamed(name="CacheLayer"))
    if isinstance(root, TypeList):
        item = root.item
        if isinstance(item, TypeOptional):
            return item
        return TypeOptional(inner=item)
    return past_type


def _check_primitive_signature(callee: str, args_count: int, kwargs: dict[str, Any]) -> None:
    op_name = _canonical_primitive_name(callee)
    signature = get_op_lowering_signature(op_name)
    if not isinstance(signature, dict):
        return
    arity = signature.get("arity")
    if arity is not None:
        min_args, max_args = arity
        if args_count < min_args or args_count > max_args:
            raise ValueError(
                f"{op_name} expects {min_args}"
                + (f"..{max_args}" if min_args != max_args else "")
                + f" positional args, got {args_count}"
            )
    allowed = signature.get("allowed_kwargs")
    if isinstance(allowed, set):
        unknown = sorted(set(kwargs) - allowed)
        if unknown:
            allowed_text = ", ".join(sorted(str(name) for name in allowed))
            if allowed_text:
                raise ValueError(
                    f"{op_name} unsupported kwargs: {', '.join(unknown)}; allowed: {allowed_text}"
                )
            raise ValueError(f"{op_name} unsupported kwargs: {', '.join(unknown)}")
    kinds = signature.get("kwarg_kinds")
    if isinstance(kinds, dict):
        for key, value in kwargs.items():
            expected = kinds.get(key)
            if expected is None:
                continue
            if not _kwarg_matches_kind(value, expected):
                raise ValueError(f"{op_name} kwarg {key!r} expects {expected}")


def _check_obsolete_call_syntax(callee: str) -> None:
    op_name = callee.split("@", 1)[0]
    if op_name == "reshape_heads_triplet":
        raise ValueError(
            "obsolete compatibility call 'reshape_heads_triplet'; "
            "call reshape_heads on q/k/v individually"
        )
    if callee.startswith("_act_"):
        raise ValueError(f"obsolete activation primitive {callee!r}; use _activations_<kind>")
    if callee.startswith("activations_"):
        raise ValueError(f"obsolete activation call {callee!r}; use _activations_<kind>")
    if "::" not in callee:
        return
    if callee.startswith("act::"):
        raise ValueError(
            f"obsolete call syntax {callee!r}; use _activations_<kind> primitive calls"
        )
    if callee.startswith("cache::"):
        raise ValueError(f"obsolete call syntax {callee!r}; use _cache_update/_cache_seq_len")
    raise ValueError(f"obsolete namespaced call syntax {callee!r}; '::' is not supported in calls")


def _canonical_primitive_name(callee: str) -> str:
    base = callee.split("@", 1)[0] if "@" in callee else callee
    if base.startswith("_cache_"):
        suffix = base[len("_cache_") :]
        if suffix == "update":
            return "cache_update"
        if suffix == "seq_len":
            return "cache_seq_len"
    if base.startswith("_") and len(base) > 1 and base[1].isalpha():
        return _canonical_primitive_name(base[1:])
    return base


def _primitive_output_type(
    *,
    callee: str,
    args: tuple[AxonExpr, ...],
    kwargs: dict[str, AxonKwargValue],
    env: dict[str, TypeExpr],
    signatures: dict[str, ModuleSignature],
    primitive_aliases: dict[str, tuple[str, int]],
    module: AxonModule,
    path: tuple[int, ...],
    dim_symbols: dict[str, DimToken],
    rigid_symbols: set[str],
    expected_arity: int | None,
) -> TypeExpr:
    def _is_generic_tensor_decl(tp: TypeExpr) -> bool:
        return isinstance(tp, TypeNamed) and tp.name in {"Tensor", "IdxTensor"}

    generic_tensor_inference_ops = {
        "reshape_heads",
        "repeat",
        "merge_heads",
        "rope_pair",
        "moe_scatter_add",
    }

    op_name = _canonical_primitive_name(callee)
    structural_ops = {"cache_update", "split", "list_append", "list_index"}
    dynamic_from_first_arg = False
    declared_return_items: tuple[TypeExpr, ...] = ()
    op_type_signature = get_op_lowering_type_signature(op_name)
    if isinstance(op_type_signature, dict):
        returns_spec = op_type_signature.get("returns")
        if isinstance(returns_spec, tuple) and all(isinstance(item, str) for item in returns_spec):
            declared_return_items = tuple(parse_type_expr(item) for item in returns_spec)
            if (
                declared_return_items
                and all(
                    not _is_any_type(tp)
                    and not (
                        _is_generic_tensor_decl(tp) and op_name in generic_tensor_inference_ops
                    )
                    for tp in declared_return_items
                )
                and op_name not in structural_ops
            ):
                if len(declared_return_items) == 1:
                    return declared_return_items[0]
                return TypeTuple(items=declared_return_items)
        if isinstance(returns_spec, str):
            if returns_spec == "dynamic":
                dynamic_from_first_arg = True
            else:
                parsed = parse_type_expr(returns_spec)
                if (
                    not _is_any_type(parsed)
                    and not (
                        _is_generic_tensor_decl(parsed) and op_name in generic_tensor_inference_ops
                    )
                    and op_name not in structural_ops
                ):
                    return parsed

    arg_types = [
        _infer_expr_type(
            arg,
            env=env,
            signatures=signatures,
            primitive_aliases=primitive_aliases,
            module=module,
            path=(*path, i),
            dim_symbols=dim_symbols,
            rigid_symbols=rigid_symbols,
        )
        for i, arg in enumerate(args)
    ]
    if op_name == "cache_update":
        present = _cache_update_present_type(arg_types[0]) if arg_types else TypeAny()
        k_ctx = (
            arg_types[1]
            if len(arg_types) > 1 and not _is_any_type(arg_types[1])
            else TypeNamed(name="Tensor")
        )
        v_ctx = (
            arg_types[2]
            if len(arg_types) > 2 and not _is_any_type(arg_types[2])
            else TypeNamed(name="Tensor")
        )
        return TypeTuple(items=(k_ctx, v_ctx, present))
    if op_name in {"split", "chunk"}:
        tensor_type = arg_types[0] if arg_types else TypeAny()
        return TypeList(item=tensor_type)
    if op_name == "list_append":
        list_arg = arg_types[0] if arg_types else TypeAny()
        if isinstance(list_arg, TypeOptional):
            raise _error(
                module,
                path,
                f"list_append expects non-optional List[_], got {render_type(list_arg)}",
            )
        list_root = list_arg
        if isinstance(list_root, TypeNamed):
            alias_item = _named_list_item_type(list_root.name)
            if alias_item is not None:
                list_root = TypeList(item=alias_item)
        if not isinstance(list_root, TypeList):
            raise _error(
                module,
                path,
                f"list_append expects first arg List[_], got {render_type(list_arg)}",
            )
        item_arg = arg_types[1] if len(arg_types) > 1 else TypeAny()
        merged_item = _unify_list_item_types(
            list_root.item,
            item_arg,
            dim_symbols=dim_symbols,
            rigid_symbols=rigid_symbols,
        )
        if merged_item is None:
            raise _error(
                module,
                path,
                "list_append item type mismatch: "
                f"list has {render_type(list_root.item)}, appended {render_type(item_arg)}",
            )
        result = TypeList(item=merged_item)
        return result
    if op_name == "list_index":
        list_arg = arg_types[0] if arg_types else TypeAny()
        if isinstance(list_arg, TypeOptional):
            raise _error(
                module,
                path,
                f"list_index expects non-optional List[_], got {render_type(list_arg)}",
            )
        list_root = list_arg
        if isinstance(list_root, TypeNamed):
            alias_item = _named_list_item_type(list_root.name)
            if alias_item is not None:
                list_root = TypeList(item=alias_item)
        if not isinstance(list_root, TypeList):
            raise _error(
                module,
                path,
                f"list_index expects first arg List[_], got {render_type(list_arg)}",
            )
        return list_root.item
    var_types: dict[str, str] = {}
    input_slots: list[tuple[str, set[str]]] = []
    slot_names = ("x", "y", "z", "a", "b", "c", "d")
    for idx, arg_type in enumerate(arg_types):
        ref = f"arg_{idx}"
        var_types[ref] = render_type(arg_type)
        slot = slot_names[idx] if idx < len(slot_names) else f"arg{idx}"
        input_slots.append((slot, {ref}))

    kwarg_scalars: dict[str, Any] = {}
    for key, value in kwargs.items():
        if isinstance(value, AxonExpr):
            scalar_value = _expr_scalar_value(value, env=env, dim_symbols=dim_symbols)
            if scalar_value is not _MISSING:
                kwarg_scalars[key] = scalar_value
                continue
            dim_token = _infer_dim_token_from_expr(value, env=env, dim_symbols=dim_symbols)
            if dim_token is not None:
                kwarg_scalars[key] = render_dim_token(dim_token)
                continue
            lit = _infer_literal_expr_type(value)
            if isinstance(lit, TypeInt):
                kwarg_scalars[key] = value.value if isinstance(value, AxonExprInt) else None
            elif isinstance(lit, TypeFloat):
                if isinstance(value, AxonExprFloat):
                    kwarg_scalars[key] = value.lexeme if value.lexeme else value.value
            elif isinstance(lit, TypeBool):
                kwarg_scalars[key] = value.value if isinstance(value, AxonExprBool) else None
            elif isinstance(lit, TypeNull):
                kwarg_scalars[key] = None
            elif isinstance(lit, TypeString):
                kwarg_scalars[key] = value.value if isinstance(value, AxonExprString) else None
            else:
                inferred_type = _infer_expr_type(
                    value,
                    env=env,
                    signatures=signatures,
                    primitive_aliases=primitive_aliases,
                    module=module,
                    path=path,
                    dim_symbols=dim_symbols,
                    rigid_symbols=rigid_symbols,
                )
                if isinstance(inferred_type, TypeInt):
                    kwarg_scalars[key] = 0
                elif isinstance(inferred_type, TypeFloat):
                    kwarg_scalars[key] = 0.0
                elif isinstance(inferred_type, TypeBool):
                    kwarg_scalars[key] = False
                elif isinstance(inferred_type, TypeString):
                    kwarg_scalars[key] = ""
                elif isinstance(inferred_type, TypeList):
                    kwarg_scalars[key] = _list_type_placeholder(inferred_type.item)
                else:
                    kwarg_scalars[key] = None
            continue
        kwarg_scalars[key] = value

    known_arity_fn = get_op_lowering_known_output_arity(op_name)
    arity = known_arity_fn(kwargs=kwarg_scalars) if callable(known_arity_fn) else None
    if isinstance(arity, int) and arity > 0:
        out_arity = arity
    elif len(declared_return_items) > 1:
        out_arity = len(declared_return_items)
    else:
        out_arity = 1
    output_names = [f"out_{i}" for i in range(out_arity)]
    inferred_outputs = infer_output_types_for_node(
        op_name=op_name,
        node_spec=kwarg_scalars,
        input_slots=input_slots,
        output_vars=output_names,
        var_types=var_types,
    )
    fallback = _primitive_fallback_type(op_name, arg_types=arg_types, arity=out_arity)
    if out_arity > 1:
        fallback_items: tuple[TypeExpr, ...] = ()
        if isinstance(fallback, TypeTuple):
            fallback_items = fallback.items
        elif declared_return_items:
            fallback_items = declared_return_items
        items: list[TypeExpr] = []
        for idx, out in enumerate(output_names):
            token = inferred_outputs.get(out)
            if isinstance(token, str) and token:
                items.append(parse_type_expr(token))
                continue
            if idx < len(fallback_items) and not _is_any_type(fallback_items[idx]):
                items.append(fallback_items[idx])
                continue
            if dynamic_from_first_arg and arg_types:
                items.append(arg_types[0])
                continue
            items.append(TypeAny())
        return TypeTuple(items=tuple(items))
    token = inferred_outputs.get(output_names[0])
    if isinstance(token, str) and token:
        return parse_type_expr(token)
    if dynamic_from_first_arg and arg_types:
        return arg_types[0]
    if len(declared_return_items) == 1 and not _is_any_type(declared_return_items[0]):
        return declared_return_items[0]
    if isinstance(fallback, _TYPE_EXPR_CLASSES):
        return fallback
    return TypeAny()


def _infer_dim_token_from_expr(
    expr: AxonExpr,
    *,
    env: dict[str, TypeExpr],
    dim_symbols: dict[str, DimToken] | None = None,
) -> DimToken | None:
    if isinstance(expr, AxonExprInt):
        return expr.value
    if isinstance(expr, AxonExprName):
        if expr.name in env:
            return expr.name
        if isinstance(dim_symbols, dict) and expr.name in dim_symbols:
            return expr.name
        return None
    if isinstance(expr, AxonExprParen):
        return _infer_dim_token_from_expr(expr.inner, env=env, dim_symbols=dim_symbols)
    if isinstance(expr, AxonExprBinary) and expr.op in {"+", "-", "*", "/"}:
        left = _infer_dim_token_from_expr(expr.left, env=env, dim_symbols=dim_symbols)
        right = _infer_dim_token_from_expr(expr.right, env=env, dim_symbols=dim_symbols)
        if left is None or right is None:
            return None
        return DimExprBinary(op=expr.op, left=left, right=right)
    return None


def _infer_literal_expr_type(expr: AxonExpr) -> TypeExpr | None:
    if isinstance(expr, AxonExprInt):
        return TypeInt()
    if isinstance(expr, AxonExprFloat):
        return TypeFloat()
    if isinstance(expr, AxonExprBool):
        return TypeBool()
    if isinstance(expr, AxonExprNull):
        return TypeNull()
    if isinstance(expr, AxonExprString):
        return TypeString()
    return None


_MISSING = object()


def _list_type_placeholder(item_type: TypeExpr) -> list[Any]:
    root = item_type.inner if isinstance(item_type, TypeOptional) else item_type
    if isinstance(root, TypeInt):
        return [0]
    if isinstance(root, TypeFloat):
        return [0.0]
    if isinstance(root, TypeString):
        return [""]
    if isinstance(root, TypeBool):
        return [False]
    if isinstance(root, TypeNamed):
        if root.name in {"Tensor", "IdxTensor"}:
            return [root.name]
        return [root.name]
    return []


def _expr_scalar_value(
    expr: AxonExpr,
    *,
    env: dict[str, TypeExpr],
    dim_symbols: dict[str, DimToken] | None = None,
) -> Any:
    if isinstance(expr, AxonExprInt):
        return expr.value
    if isinstance(expr, AxonExprFloat):
        return expr.lexeme if expr.lexeme else expr.value
    if isinstance(expr, AxonExprBool):
        return expr.value
    if isinstance(expr, AxonExprNull):
        return None
    if isinstance(expr, AxonExprString):
        return expr.value
    if isinstance(expr, AxonExprName):
        return expr.name
    if isinstance(expr, AxonExprParen):
        return _expr_scalar_value(expr.inner, env=env, dim_symbols=dim_symbols)
    if isinstance(expr, AxonExprList):
        items: list[Any] = []
        for item in expr.items:
            value = _expr_scalar_value(item, env=env, dim_symbols=dim_symbols)
            if value is _MISSING:
                dim_token = _infer_dim_token_from_expr(item, env=env, dim_symbols=dim_symbols)
                if dim_token is None:
                    return _MISSING
                value = render_dim_token(dim_token)
            items.append(value)
        return items
    dim_token = _infer_dim_token_from_expr(expr, env=env, dim_symbols=dim_symbols)
    if dim_token is not None:
        return render_dim_token(dim_token)
    return _MISSING


def _is_tensor_like(tp: TypeExpr) -> bool:
    return isinstance(tp, TypeTensor) or (
        isinstance(tp, TypeNamed) and tp.name in {"Tensor", "IdxTensor"}
    )


def _tensor_like_binary_result(
    left: TypeExpr,
    right: TypeExpr,
    *,
    dim_symbols: dict[str, DimToken],
    rigid_symbols: set[str],
) -> TypeExpr | None:
    if not (_is_tensor_like(left) or _is_tensor_like(right)):
        return None
    if _is_tensor_like(left) and _is_tensor_like(right):
        if isinstance(left, TypeTensor) and isinstance(right, TypeTensor):
            broadcasted = _broadcast_dims(
                left.dims,
                right.dims,
                symbols=dim_symbols,
                rigid_symbols=rigid_symbols,
            )
            if broadcasted is None:
                return None
            return TypeTensor(base=left.base, dims=broadcasted)
        if isinstance(left, TypeTensor):
            return left
        if isinstance(right, TypeTensor):
            return right
        return TypeNamed(name="Tensor")
    if _is_tensor_like(left) and is_numeric_type(right):
        return left
    if _is_tensor_like(right) and is_numeric_type(left):
        return right
    return None


def _broadcast_dims(
    left: tuple[DimToken, ...],
    right: tuple[DimToken, ...],
    *,
    symbols: dict[str, DimToken],
    rigid_symbols: set[str],
) -> tuple[DimToken, ...] | None:
    max_rank = max(len(left), len(right))
    left_full = (1,) * (max_rank - len(left)) + left
    right_full = (1,) * (max_rank - len(right)) + right
    result: list[DimToken] = []
    subst: dict[str, DimToken] = {}
    for left_raw, right_raw in zip(left_full, right_full, strict=True):
        left_dim = _normalize_dim(left_raw, subst=subst, symbols=symbols)
        right_dim = _normalize_dim(right_raw, subst=subst, symbols=symbols)
        if _is_flexible_dim_var(right_dim, rigid_symbols):
            assert isinstance(right_dim, str)
            prev = subst.get(right_dim)
            if prev is None:
                subst[right_dim] = left_dim
                right_dim = left_dim
            else:
                if not _dim_equal(prev, left_dim, subst=subst, symbols=symbols):
                    return None
                right_dim = prev
        if _is_flexible_dim_var(left_dim, rigid_symbols):
            assert isinstance(left_dim, str)
            prev = subst.get(left_dim)
            if prev is None:
                subst[left_dim] = right_dim
                left_dim = right_dim
            else:
                if not _dim_equal(prev, right_dim, subst=subst, symbols=symbols):
                    return None
                left_dim = prev
        if _dim_equal(left_dim, right_dim, subst=subst, symbols=symbols):
            result.append(left_dim)
            continue
        if left_dim == 1:
            result.append(right_dim)
            continue
        if right_dim == 1:
            result.append(left_dim)
            continue
        return None
    return tuple(result)


def _expr_name(expr: AxonExpr) -> str | None:
    if isinstance(expr, AxonExprName):
        return expr.name
    return None


def _arity_kwarg_value(value: Any) -> Any:
    if (
        isinstance(value, tuple)
        and len(value) == 3
        and value[0] == "__expr__"
        and isinstance(value[1], _TYPE_EXPR_CLASSES)
    ):
        inferred = value[1]
        dim_token = value[2] if isinstance(value[2], int | str | DimExprBinary) else None
        if dim_token is not None:
            return render_dim_token(dim_token)
        if isinstance(inferred, TypeInt):
            return 0
        if isinstance(inferred, TypeFloat):
            return 0.0
        if isinstance(inferred, TypeBool):
            return False
        if isinstance(inferred, TypeString):
            return ""
        if isinstance(inferred, TypeList):
            return _list_type_placeholder(inferred.item)
        return None
    return value


def _kwarg_value_to_expr(value: AxonKwargValue) -> AxonExpr:
    if isinstance(value, AxonExpr):
        return value
    if isinstance(value, bool):
        return AxonExprBool(value=value)
    if isinstance(value, int):
        return AxonExprInt(value=value)
    if isinstance(value, float):
        return AxonExprFloat(value=value)
    if value is None:
        return AxonExprNull()
    if isinstance(value, str):
        return AxonExprString(value=value)
    if isinstance(value, list):
        return AxonExprList(items=tuple(_kwarg_value_to_expr(item) for item in value))
    raise ValueError(f"unsupported kwarg value type for module call: {type(value).__name__}")


def _substitute_expr(expr: AxonExpr, var_name: str, replacement: AxonExpr) -> AxonExpr:
    if isinstance(expr, AxonExprName):
        if expr.name == var_name:
            return replacement
        return expr
    if isinstance(
        expr,
        AxonExprInt | AxonExprFloat | AxonExprBool | AxonExprNull | AxonExprString,
    ):
        return expr
    if isinstance(expr, AxonExprTuple):
        return AxonExprTuple(
            items=tuple(_substitute_expr(item, var_name, replacement) for item in expr.items)
        )
    if isinstance(expr, AxonExprList):
        return AxonExprList(
            items=tuple(_substitute_expr(item, var_name, replacement) for item in expr.items)
        )
    if isinstance(expr, AxonExprCall):
        new_kwargs: dict[str, AxonKwargValue] = {}
        for key, value in expr.kwargs.items():
            if isinstance(value, AxonExpr):
                new_kwargs[key] = _substitute_expr(value, var_name, replacement)
            else:
                new_kwargs[key] = value
        return AxonExprCall(
            callee=expr.callee,
            args=tuple(_substitute_expr(arg, var_name, replacement) for arg in expr.args),
            kwargs=new_kwargs,
        )
    if isinstance(expr, AxonExprPipe):
        return AxonExprPipe(
            value=_substitute_expr(expr.value, var_name, replacement),
            stages=tuple(_substitute_expr(stage, var_name, replacement) for stage in expr.stages),
        )
    if isinstance(expr, AxonExprBind):
        if expr.var == var_name:
            return AxonExprBind(
                value=_substitute_expr(expr.value, var_name, replacement),
                var=expr.var,
                body=expr.body,
            )
        return AxonExprBind(
            value=_substitute_expr(expr.value, var_name, replacement),
            var=expr.var,
            body=_substitute_expr(expr.body, var_name, replacement),
        )
    if isinstance(expr, AxonExprIf):
        return AxonExprIf(
            cond=_substitute_expr(expr.cond, var_name, replacement),
            true_expr=_substitute_expr(expr.true_expr, var_name, replacement),
            false_expr=_substitute_expr(expr.false_expr, var_name, replacement),
        )
    if isinstance(expr, AxonExprTernary):
        return AxonExprTernary(
            cond=_substitute_expr(expr.cond, var_name, replacement),
            true_expr=_substitute_expr(expr.true_expr, var_name, replacement),
            false_expr=_substitute_expr(expr.false_expr, var_name, replacement),
        )
    if isinstance(expr, AxonExprBinary):
        return AxonExprBinary(
            op=expr.op,
            left=_substitute_expr(expr.left, var_name, replacement),
            right=_substitute_expr(expr.right, var_name, replacement),
        )
    if isinstance(expr, AxonExprLambda):
        if expr.var == var_name:
            return expr
        return AxonExprLambda(var=expr.var, body=_substitute_expr(expr.body, var_name, replacement))
    if isinstance(expr, AxonExprParen):
        return AxonExprParen(inner=_substitute_expr(expr.inner, var_name, replacement))
    if isinstance(expr, AxonExprDo):
        return expr
    return expr


def _expr_name_refs(expr: AxonExpr) -> set[str]:
    out: set[str] = set()
    stack: list[AxonExpr] = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, AxonExprName):
            out.add(node.name)
            continue
        if isinstance(
            node,
            AxonExprInt | AxonExprFloat | AxonExprBool | AxonExprNull | AxonExprString,
        ):
            continue
        if isinstance(node, AxonExprParen):
            stack.append(node.inner)
            continue
        if isinstance(node, AxonExprTuple):
            stack.extend(node.items)
            continue
        if isinstance(node, AxonExprList):
            stack.extend(node.items)
            continue
        if isinstance(node, AxonExprPipe):
            stack.append(node.value)
            stack.extend(node.stages)
            continue
        if isinstance(node, AxonExprBind):
            stack.append(node.value)
            stack.append(node.body)
            continue
        if isinstance(node, AxonExprIf | AxonExprTernary):
            stack.append(node.cond)
            stack.append(node.true_expr)
            stack.append(node.false_expr)
            continue
        if isinstance(node, AxonExprBinary):
            stack.append(node.left)
            stack.append(node.right)
            continue
        if isinstance(node, AxonExprCall):
            stack.extend(node.args)
            for value in node.kwargs.values():
                if isinstance(value, AxonExpr):
                    stack.append(value)
            continue
        if isinstance(node, AxonExprLambda):
            stack.append(node.body)
            continue
    return out


def _call_return_type(
    *,
    callee: str,
    args: tuple[AxonExpr, ...],
    kwargs: dict[str, AxonKwargValue],
    env: dict[str, TypeExpr],
    signatures: dict[str, ModuleSignature],
    primitive_aliases: dict[str, tuple[str, int]],
    module: AxonModule,
    path: tuple[int, ...],
    dim_symbols: dict[str, DimToken],
    rigid_symbols: set[str],
    expected_arity: int | None = None,
) -> TypeExpr:
    raw_callee = callee
    if "@@" in callee:
        callee = callee.replace("@@", "@", 1)

    def _apply_primitive_alias(name: str) -> str:
        alias_parts = name.split("@")
        alias_base_name = alias_parts[0]
        alias_path_parts = alias_parts[1:]
        alias = primitive_aliases.get(alias_base_name)
        if alias is None:
            return name
        alias_base, expected_path_count = alias
        if expected_path_count != len(alias_path_parts):
            return name
        return "@".join([alias_base, *alias_path_parts]) if alias_path_parts else alias_base

    def _resolve_unqualified_import_member(name: str) -> str:
        base_name, *path_suffix = name.split("@")
        if "." in base_name or base_name in signatures:
            return name

        explicit_namespaces: list[str] = []
        if isinstance(module.imported_members, dict):
            for namespace, members in module.imported_members.items():
                if base_name in members and f"{namespace}.{base_name}" in signatures:
                    explicit_namespaces.append(namespace)
        if explicit_namespaces:
            if len(explicit_namespaces) > 1:
                choices = ", ".join(sorted(explicit_namespaces))
                raise _error(
                    module,
                    path,
                    f"ambiguous imported member {base_name!r}; found in namespaces: {choices}",
                )
            qualified_base = f"{explicit_namespaces[0]}.{base_name}"
            return "@".join([qualified_base, *path_suffix]) if path_suffix else qualified_base

        prelude_qualified = f"Prelude.{base_name}"
        if prelude_qualified in signatures:
            return "@".join([prelude_qualified, *path_suffix]) if path_suffix else prelude_qualified
        return name

    callee = _apply_primitive_alias(callee)
    callee = _resolve_unqualified_import_member(callee)
    callee = _apply_primitive_alias(callee)
    base = callee.split("@", 1)[0]
    implicit_activation = _IMPLICIT_ACTIVATION_ALIASES.get(base)
    if implicit_activation is not None and "@" not in callee:
        callee = implicit_activation
    _check_obsolete_call_syntax(callee)
    call_sig = signatures.get(callee)
    if call_sig is None and "@" in callee:
        callee_parts = callee.split("@")
        callee_base = callee_parts[0]
        callee_paths = callee_parts[1:]
        base_sig = signatures.get(callee_base)
        if base_sig is not None and base_sig.path_param_count == len(callee_paths):
            call_sig = base_sig
            callee = callee_base
    if call_sig is None:
        callee_parts = callee.split("@")
        callee_base = callee_parts[0]
        callee_paths = callee_parts[1:]
        if "." in callee_base:
            member_base = callee_base.rsplit(".", 1)[1]
            member_sig = signatures.get(member_base)
            if member_sig is not None and member_sig.path_param_count == len(callee_paths):
                call_sig = member_sig
                callee = member_base
    if call_sig is None:
        callee_parts = callee.split("@")
        callee_base = callee_parts[0]
        callee_paths = callee_parts[1:]
        if "." not in callee_base:
            candidates: list[tuple[str, ModuleSignature]] = []
            for module_name, module_sig in signatures.items():
                leaf = module_name.rsplit(".", 1)[-1]
                if leaf != callee_base:
                    continue
                if module_sig.path_param_count != len(callee_paths):
                    continue
                candidates.append((module_name, module_sig))
            if len(candidates) == 1:
                chosen_name, chosen_sig = candidates[0]
                call_sig = chosen_sig
                callee = chosen_name
    if call_sig is not None:
        if len(args) > len(call_sig.params):
            raise _error(
                module,
                path,
                f"call {callee!r} expects {len(call_sig.params)} args, got {len(args)}",
            )
        bound_args: list[AxonExpr | None] = [None] * len(call_sig.params)
        for idx, arg_expr in enumerate(args):
            bound_args[idx] = arg_expr
        param_index_by_name = {name: idx for idx, name in enumerate(call_sig.param_names)}
        for kw_name, kw_value in kwargs.items():
            kw_idx = param_index_by_name.get(kw_name)
            if kw_idx is None:
                raise _error(module, path, f"unknown kwarg {kw_name!r} for call {callee!r}")
            if bound_args[kw_idx] is not None:
                raise _error(
                    module,
                    path,
                    f"call {callee!r} received multiple values for argument {kw_name!r}",
                )
            bound_args[kw_idx] = _kwarg_value_to_expr(kw_value)
        missing: list[str] = []
        defaulted_param_idxs: set[int] = set()
        param_name_set = set(call_sig.param_names)
        for idx, bound_value in enumerate(bound_args):
            if bound_value is not None:
                continue
            default_expr = (
                call_sig.param_default_exprs[idx]
                if idx < len(call_sig.param_default_exprs)
                else None
            )
            if isinstance(default_expr, AxonExpr):
                resolved_default = default_expr
                for sub_name, sub_value in zip(
                    call_sig.param_names,
                    bound_args,
                    strict=True,
                ):
                    if not isinstance(sub_value, AxonExpr):
                        continue
                    resolved_default = _substitute_expr(resolved_default, sub_name, sub_value)
                unresolved_params = sorted(
                    {name for name in _expr_name_refs(resolved_default) if name in param_name_set}
                )
                if unresolved_params:
                    missing.extend(unresolved_params)
                    continue
                bound_args[idx] = resolved_default
                defaulted_param_idxs.add(idx)
                continue
            if idx < len(call_sig.optional_params) and call_sig.optional_params[idx]:
                bound_args[idx] = AxonExprNull()
                continue
            missing.append(call_sig.param_names[idx])
        if missing:
            raise _error(
                module,
                path,
                f"call {callee!r} missing required args: {', '.join(missing)}",
            )
        dim_subst: dict[str, DimToken] = {}
        rest_subst: dict[str, tuple[DimToken, ...]] = {}
        type_subst: dict[str, TypeExpr] = {}
        for idx, param_type in enumerate(call_sig.params):
            arg_expr_maybe = bound_args[idx]
            if arg_expr_maybe is None:
                raise _error(
                    module,
                    path,
                    f"internal typecheck error: missing bound arg {call_sig.param_names[idx]!r} for call {callee!r}",
                )
            arg_expr = arg_expr_maybe
            if idx in defaulted_param_idxs:
                continue
            param_name = call_sig.param_names[idx] if idx < len(call_sig.param_names) else None
            is_int_param = isinstance(param_type, TypeInt) or (
                isinstance(param_type, TypeOptional) and isinstance(param_type.inner, TypeInt)
            )
            if (
                is_int_param
                and param_name in {"dim", "start", "end"}
                and callee
                in {
                    "slice",
                    "Prelude.slice",
                }
            ):
                dim_token = _infer_dim_token_from_expr(arg_expr, env=env, dim_symbols=dim_symbols)
                if dim_token is not None:
                    dim_subst[param_name] = dim_token
                    continue
            if param_name == "dim" and isinstance(param_type, TypeInt):
                dim_token = _infer_dim_token_from_expr(arg_expr, env=env, dim_symbols=dim_symbols)
                if dim_token is not None:
                    dim_subst["dim"] = dim_token
                    if callee == "linear":
                        dim_subst["DO"] = dim_token
                    continue
            arg_type = _infer_expr_type(
                arg_expr,
                env=env,
                signatures=signatures,
                primitive_aliases=primitive_aliases,
                module=module,
                path=(*path, idx),
                dim_symbols=dim_symbols,
                rigid_symbols=rigid_symbols,
            )
            if isinstance(arg_type, TypeTensor) and isinstance(param_type, TypeTensor):
                if arg_type.base != param_type.base and not {
                    arg_type.base,
                    param_type.base,
                } <= {"Tensor", "IdxTensor"}:
                    raise _error(
                        module,
                        path,
                        "call type mismatch for "
                        f"{callee!r} arg {idx}: expected {render_type(param_type)}, "
                        f"got {render_type(arg_type)}",
                    )
                if not _dims_compatible(
                    arg_type.dims,
                    param_type.dims,
                    dim_subst,
                    symbols=dim_symbols,
                    rigid_symbols=rigid_symbols,
                    rest_subst=rest_subst,
                ):
                    raise _error(
                        module,
                        path,
                        f"shape mismatch in call {callee!r} for param {call_sig.param_names[idx]!r}: "
                        f"expected {param_type.dims}, got {arg_type.dims}",
                    )
                continue
            if not _types_compatible(
                arg_type,
                _substitute_type_vars(param_type, type_subst),
                dim_subst=dim_subst,
                dim_symbols=dim_symbols,
                rigid_symbols=rigid_symbols,
                rest_subst=rest_subst,
            ):
                if _bind_type_vars(
                    arg_type,
                    param_type,
                    type_subst=type_subst,
                    dim_subst=dim_subst,
                    dim_symbols=dim_symbols,
                    rigid_symbols=rigid_symbols,
                    rest_subst=rest_subst,
                ):
                    continue
                raise _error(
                    module,
                    path,
                    "call type mismatch for "
                    f"{callee!r} arg {idx}: expected {render_type(param_type)}, "
                    f"got {render_type(arg_type)}",
                )
            if param_name == "dim" and isinstance(param_type, TypeInt | TypeFloat):
                dim_token = _infer_dim_token_from_expr(arg_expr, env=env, dim_symbols=dim_symbols)
                if dim_token is not None:
                    dim_subst[param_name] = dim_token
                    if callee == "linear":
                        dim_subst["DO"] = dim_token
        substituted_returns = tuple(
            _substitute_type_dims(
                _substitute_type_vars(ret, type_subst),
                dim_subst=dim_subst,
                symbols=dim_symbols,
                rest_subst=rest_subst,
            )
            for ret in call_sig.returns
        )
        if len(substituted_returns) == 1:
            return substituted_returns[0]
        return TypeTuple(items=substituted_returns)

    if callee in {"sqrt", "Prelude.sqrt"}:
        if kwargs:
            raise _error(module, path, f"{callee!r} does not support kwargs")
        if len(args) != 1:
            raise _error(module, path, f"{callee!r} expects exactly one positional argument")
        arg_type = _infer_expr_type(
            args[0],
            env=env,
            signatures=signatures,
            primitive_aliases=primitive_aliases,
            module=module,
            path=(*path, 0),
            dim_symbols=dim_symbols,
            rigid_symbols=rigid_symbols,
        )
        if not isinstance(arg_type, TypeInt | TypeFloat):
            raise _error(
                module,
                path,
                f"{callee!r} expects numeric argument, got {render_type(arg_type)}",
            )
        return TypeFloat()

    raw_base = raw_callee.split("@", 1)[0].strip()
    if raw_base == "_linear" and kwargs:
        raise _error(
            module,
            path,
            "_linear only accepts positional arguments; use Prelude.linear for keyword/default syntax",
        )
    if raw_base == "_layernorm" and kwargs:
        raise _error(
            module,
            path,
            "_layernorm only accepts positional arguments; use Prelude.layernorm for keyword/default syntax",
        )
    if raw_base == "_embedding" and kwargs:
        raise _error(
            module,
            path,
            "_embedding only accepts positional arguments; use Prelude.embedding for keyword/default syntax",
        )
    if raw_base == "_split" and kwargs:
        raise _error(
            module,
            path,
            "_split only accepts positional arguments; use Prelude.split for keyword/default syntax",
        )
    if raw_base == "_cast" and kwargs:
        raise _error(
            module,
            path,
            "_cast only accepts positional arguments; use Prelude.cast for keyword/default syntax",
        )
    if raw_base == "_cumsum" and kwargs:
        raise _error(
            module,
            path,
            "_cumsum only accepts positional arguments; use Prelude.cumsum for keyword/default syntax",
        )
    if raw_base == "_arange" and kwargs:
        raise _error(
            module,
            path,
            "_arange only accepts positional arguments; use Prelude.arange for keyword/default syntax",
        )
    if raw_base == "_sinusoidal_positions" and kwargs:
        raise _error(
            module,
            path,
            "_sinusoidal_positions only accepts positional arguments; use Prelude.sinusoidal_positions for keyword/default syntax",
        )
    if raw_base == "_expand" and kwargs:
        raise _error(
            module,
            path,
            "_expand only accepts positional arguments; use Prelude.expand for keyword/default syntax",
        )
    if raw_base == "_slice" and kwargs:
        raise _error(
            module,
            path,
            "_slice only accepts positional arguments; use Prelude.slice for keyword/default syntax",
        )

    op_name = _canonical_primitive_name(callee)
    if (
        get_op_lowering_signature(op_name) is None
        and get_op_lowering_type_signature(op_name) is None
        and get_op_lowering_known_output_arity(op_name) is None
    ):
        raise _error(module, path, f"unknown callee {callee!r}")

    signature = get_op_lowering_signature(op_name)
    kwarg_kinds = signature.get("kwarg_kinds") if isinstance(signature, dict) else {}
    rendered_kwargs: dict[str, Any] = {}
    for key, value in kwargs.items():
        expected_kind = kwarg_kinds.get(key) if isinstance(kwarg_kinds, dict) else None
        if isinstance(value, AxonExprName) and value.name in env:
            inferred_name_type = env[value.name]
            if expected_kind == "str":
                root = (
                    inferred_name_type.inner
                    if isinstance(inferred_name_type, TypeOptional)
                    else inferred_name_type
                )
                if isinstance(root, TypeString):
                    dim_token = _infer_dim_token_from_expr(value, env=env, dim_symbols=dim_symbols)
                    rendered_kwargs[key] = ("__expr__", inferred_name_type, dim_token)
                else:
                    rendered_kwargs[key] = value.name
                continue
            dim_token = _infer_dim_token_from_expr(value, env=env, dim_symbols=dim_symbols)
            rendered_kwargs[key] = ("__expr__", inferred_name_type, dim_token)
            continue
        if isinstance(value, AxonExpr):
            scalar_value = _expr_scalar_value(value, env=env, dim_symbols=dim_symbols)
            if scalar_value is not _MISSING:
                rendered_kwargs[key] = scalar_value
                continue
            inferred = _infer_expr_type(
                value,
                env=env,
                signatures=signatures,
                primitive_aliases=primitive_aliases,
                module=module,
                path=(*path, 1000),
                dim_symbols=dim_symbols,
                rigid_symbols=rigid_symbols,
            )
            dim_token = _infer_dim_token_from_expr(value, env=env, dim_symbols=dim_symbols)
            rendered_kwargs[key] = ("__expr__", inferred, dim_token)
            continue
        rendered_kwargs[key] = value
    try:
        _check_primitive_signature(callee, len(args), rendered_kwargs)
    except ValueError as exc:
        raise _error(module, path, str(exc)) from exc
    known_arity_fn = get_op_lowering_known_output_arity(_canonical_primitive_name(callee))
    arity_kwargs = {key: _arity_kwarg_value(value) for key, value in rendered_kwargs.items()}
    if callable(known_arity_fn):
        _ = known_arity_fn(kwargs=arity_kwargs)
    return _primitive_output_type(
        callee=callee,
        args=args,
        kwargs=kwargs,
        env=env,
        signatures=signatures,
        primitive_aliases=primitive_aliases,
        module=module,
        path=path,
        dim_symbols=dim_symbols,
        rigid_symbols=rigid_symbols,
        expected_arity=expected_arity,
    )


def _infer_expr_type(
    expr: AxonExpr,
    *,
    env: dict[str, TypeExpr],
    signatures: dict[str, ModuleSignature],
    primitive_aliases: dict[str, tuple[str, int]],
    module: AxonModule,
    path: tuple[int, ...],
    dim_symbols: dict[str, DimToken],
    rigid_symbols: set[str],
    expected_arity: int | None = None,
) -> TypeExpr:
    literal = _infer_literal_expr_type(expr)
    if literal is not None:
        return literal
    if isinstance(expr, AxonExprName):
        if expr.name in env:
            return env[expr.name]
        return TypeAny()
    if isinstance(expr, AxonExprList):
        if not expr.items:
            return TypeList(item=TypeAny())
        item_types = [
            _infer_expr_type(
                item,
                env=env,
                signatures=signatures,
                primitive_aliases=primitive_aliases,
                module=module,
                path=(*path, i),
                dim_symbols=dim_symbols,
                rigid_symbols=rigid_symbols,
            )
            for i, item in enumerate(expr.items)
        ]
        unified = item_types[0]
        for item_type in item_types[1:]:
            merged = _unify_list_item_types(
                unified,
                item_type,
                dim_symbols=dim_symbols,
                rigid_symbols=rigid_symbols,
            )
            if merged is None:
                raise _error(
                    module,
                    path,
                    "list literal has incompatible element types: "
                    f"{render_type(unified)} vs {render_type(item_type)}",
                )
            unified = merged
        return TypeList(item=unified)
    if isinstance(expr, AxonExprTuple):
        return TypeTuple(
            items=tuple(
                _infer_expr_type(
                    item,
                    env=env,
                    signatures=signatures,
                    primitive_aliases=primitive_aliases,
                    module=module,
                    path=(*path, i),
                    dim_symbols=dim_symbols,
                    rigid_symbols=rigid_symbols,
                )
                for i, item in enumerate(expr.items)
            )
        )
    if isinstance(expr, AxonExprParen):
        return _infer_expr_type(
            expr.inner,
            env=env,
            signatures=signatures,
            primitive_aliases=primitive_aliases,
            module=module,
            path=path,
            dim_symbols=dim_symbols,
            rigid_symbols=rigid_symbols,
        )
    if isinstance(expr, AxonExprLambda):
        return TypeAny()
    if isinstance(expr, AxonExprCall):
        return _call_return_type(
            callee=expr.callee,
            args=expr.args,
            kwargs=expr.kwargs,
            env=env,
            signatures=signatures,
            primitive_aliases=primitive_aliases,
            module=module,
            path=path,
            dim_symbols=dim_symbols,
            rigid_symbols=rigid_symbols,
            expected_arity=expected_arity,
        )
    if isinstance(expr, AxonExprDo):
        nested_env = dict(env)
        _typecheck_statements(
            expr.body,
            env=nested_env,
            signatures=signatures,
            primitive_aliases=primitive_aliases,
            module=module,
            declared_returns=(),
            path=path,
            dim_symbols=dim_symbols,
            rigid_symbols=rigid_symbols,
        )
        return_types = _collect_return_types(
            expr.body,
            env=nested_env,
            signatures=signatures,
            primitive_aliases=primitive_aliases,
            module=module,
            path=path,
            dim_symbols=dim_symbols,
            rigid_symbols=rigid_symbols,
        )
        if not return_types:
            raise _error(module, path, "do expression has no typed return path")
        first = return_types[0]
        if len(first) == 1:
            return first[0]
        return TypeTuple(items=first)
    if isinstance(expr, AxonExprPipe):
        current: TypeExpr = _infer_expr_type(
            expr.value,
            env=env,
            signatures=signatures,
            primitive_aliases=primitive_aliases,
            module=module,
            path=(*path, 0),
            dim_symbols=dim_symbols,
            rigid_symbols=rigid_symbols,
        )
        for stage_idx, stage in enumerate(expr.stages, start=1):
            pipe_names: list[str]
            tmp_env = dict(env)
            if isinstance(current, TypeTuple):
                pipe_names = [f"__pipe_{stage_idx}_{i}" for i in range(len(current.items))]
                for name, item_type in zip(pipe_names, current.items, strict=True):
                    tmp_env[name] = item_type
            else:
                if stage_idx == 1 and isinstance(expr.value, AxonExprName):
                    pipe_names = [expr.value.name]
                else:
                    pipe_names = [f"__pipe_{stage_idx}"]
                    tmp_env[pipe_names[0]] = current

            if isinstance(stage, AxonExprCall):
                stage_args = list(stage.args)
                if stage_args and len(stage_args) >= len(pipe_names):
                    same_prefix = True
                    for idx_ref, ref in enumerate(pipe_names):
                        head = stage_args[idx_ref]
                        if not isinstance(head, AxonExprName) or head.name != ref:
                            same_prefix = False
                            break
                    if same_prefix:
                        stage_args = stage_args[len(pipe_names) :]
                stage_expr = AxonExprCall(
                    callee=stage.callee,
                    args=tuple([*(AxonExprName(name=name) for name in pipe_names), *stage_args]),
                    kwargs=stage.kwargs,
                )
                current = _infer_expr_type(
                    stage_expr,
                    env=tmp_env,
                    signatures=signatures,
                    primitive_aliases=primitive_aliases,
                    module=module,
                    path=(*path, stage_idx),
                    dim_symbols=dim_symbols,
                    rigid_symbols=rigid_symbols,
                )
                continue
            if isinstance(stage, AxonExprName):
                stage_expr = AxonExprCall(
                    callee=stage.name,
                    args=tuple(AxonExprName(name=name) for name in pipe_names),
                    kwargs={},
                )
                current = _infer_expr_type(
                    stage_expr,
                    env=tmp_env,
                    signatures=signatures,
                    primitive_aliases=primitive_aliases,
                    module=module,
                    path=(*path, stage_idx),
                    dim_symbols=dim_symbols,
                    rigid_symbols=rigid_symbols,
                )
                continue
            current = _infer_expr_type(
                stage,
                env=env,
                signatures=signatures,
                primitive_aliases=primitive_aliases,
                module=module,
                path=(*path, stage_idx),
                dim_symbols=dim_symbols,
                rigid_symbols=rigid_symbols,
            )
        return current
    if isinstance(expr, AxonExprBind):
        left_type = _infer_expr_type(
            expr.value,
            env=env,
            signatures=signatures,
            primitive_aliases=primitive_aliases,
            module=module,
            path=(*path, 0),
            dim_symbols=dim_symbols,
            rigid_symbols=rigid_symbols,
        )
        inner_env = dict(env)
        inner_env[expr.var] = left_type
        return _infer_expr_type(
            expr.body,
            env=inner_env,
            signatures=signatures,
            primitive_aliases=primitive_aliases,
            module=module,
            path=(*path, 1),
            dim_symbols=dim_symbols,
            rigid_symbols=rigid_symbols,
        )
    if isinstance(expr, AxonExprIf | AxonExprTernary):
        cond_type = _infer_expr_type(
            expr.cond,
            env=env,
            signatures=signatures,
            primitive_aliases=primitive_aliases,
            module=module,
            path=(*path, 0),
            dim_symbols=dim_symbols,
            rigid_symbols=rigid_symbols,
        )
        if not is_bool_like(cond_type):
            raise _error(
                module,
                path,
                f"condition must be Bool-compatible, got {render_type(cond_type)}",
            )
        true_env, false_env = _refine_env_for_null_test(expr.cond, env)
        true_type = _infer_expr_type(
            expr.true_expr,
            env=true_env,
            signatures=signatures,
            primitive_aliases=primitive_aliases,
            module=module,
            path=(*path, 1),
            dim_symbols=dim_symbols,
            rigid_symbols=rigid_symbols,
        )
        false_type = _infer_expr_type(
            expr.false_expr,
            env=false_env,
            signatures=signatures,
            primitive_aliases=primitive_aliases,
            module=module,
            path=(*path, 2),
            dim_symbols=dim_symbols,
            rigid_symbols=rigid_symbols,
        )
        if isinstance(true_type, TypeNull) and not isinstance(false_type, TypeNull):
            return TypeOptional(inner=false_type)
        if isinstance(false_type, TypeNull) and not isinstance(true_type, TypeNull):
            return TypeOptional(inner=true_type)
        if isinstance(true_type, TypeTuple) and isinstance(false_type, TypeTuple):
            if len(true_type.items) != len(false_type.items):
                min_len = min(len(true_type.items), len(false_type.items))
                if min_len > 0:
                    true_prefix = TypeTuple(items=true_type.items[:min_len])
                    false_prefix = TypeTuple(items=false_type.items[:min_len])
                    if _types_compatible(
                        true_prefix,
                        false_prefix,
                        dim_subst={},
                        dim_symbols=dim_symbols,
                        rigid_symbols=rigid_symbols,
                    ) or _types_compatible(
                        false_prefix,
                        true_prefix,
                        dim_subst={},
                        dim_symbols=dim_symbols,
                        rigid_symbols=rigid_symbols,
                    ):
                        merged_items: list[TypeExpr] = []
                        for left_item, right_item in zip(
                            true_prefix.items, false_prefix.items, strict=True
                        ):
                            if _is_any_type(left_item) and not _is_any_type(right_item):
                                merged_items.append(right_item)
                                continue
                            merged_items.append(left_item)
                        return TypeTuple(items=tuple(merged_items))
        if _types_compatible(
            true_type,
            false_type,
            dim_subst={},
            dim_symbols=dim_symbols,
            rigid_symbols=rigid_symbols,
        ):
            return true_type
        if _types_compatible(
            false_type,
            true_type,
            dim_subst={},
            dim_symbols=dim_symbols,
            rigid_symbols=rigid_symbols,
        ):
            return false_type
        raise _error(
            module,
            path,
            f"incompatible branch types: {render_type(true_type)} vs {render_type(false_type)}",
        )
    if isinstance(expr, AxonExprBinary):
        left = _infer_expr_type(
            expr.left,
            env=env,
            signatures=signatures,
            primitive_aliases=primitive_aliases,
            module=module,
            path=(*path, 0),
            dim_symbols=dim_symbols,
            rigid_symbols=rigid_symbols,
        )
        right = _infer_expr_type(
            expr.right,
            env=env,
            signatures=signatures,
            primitive_aliases=primitive_aliases,
            module=module,
            path=(*path, 1),
            dim_symbols=dim_symbols,
            rigid_symbols=rigid_symbols,
        )
        if expr.op in {"==", "!=", "<", "<=", ">", ">=", "and", "or"}:
            tensor_result = _tensor_like_binary_result(
                left,
                right,
                dim_symbols=dim_symbols,
                rigid_symbols=rigid_symbols,
            )
            if tensor_result is not None:
                return tensor_result
            return TypeBool()
        if is_numeric_type(left) and is_numeric_type(right):
            if isinstance(left, TypeFloat) or isinstance(right, TypeFloat):
                return TypeFloat()
            return TypeInt()
        tensor_result = _tensor_like_binary_result(
            left,
            right,
            dim_symbols=dim_symbols,
            rigid_symbols=rigid_symbols,
        )
        if tensor_result is not None:
            return tensor_result
        raise _error(
            module,
            path,
            f"invalid binary operation {expr.op!r} for {render_type(left)} and {render_type(right)}",
        )
    raise _error(module, path, "unsupported expression form for type inference")


def _unify_list_item_types(
    left: TypeExpr,
    right: TypeExpr,
    *,
    dim_symbols: dict[str, DimToken],
    rigid_symbols: set[str],
) -> TypeExpr | None:
    if _is_any_type(left):
        return right
    if _is_any_type(right):
        return left
    if _types_compatible(
        left,
        right,
        dim_subst={},
        dim_symbols=dim_symbols,
        rigid_symbols=rigid_symbols,
    ) and _types_compatible(
        right,
        left,
        dim_subst={},
        dim_symbols=dim_symbols,
        rigid_symbols=rigid_symbols,
    ):
        return left
    if isinstance(left, TypeInt) and isinstance(right, TypeFloat):
        return TypeFloat()
    if isinstance(left, TypeFloat) and isinstance(right, TypeInt):
        return TypeFloat()
    if isinstance(left, TypeNull):
        return TypeOptional(inner=right)
    if isinstance(right, TypeNull):
        return TypeOptional(inner=left)
    if isinstance(left, TypeOptional):
        if _types_compatible(
            right,
            left.inner,
            dim_subst={},
            dim_symbols=dim_symbols,
            rigid_symbols=rigid_symbols,
        ):
            return left
    if isinstance(right, TypeOptional):
        if _types_compatible(
            left,
            right.inner,
            dim_subst={},
            dim_symbols=dim_symbols,
            rigid_symbols=rigid_symbols,
        ):
            return right
    return None


def _collect_return_types(
    statements: tuple[AxonStatement, ...],
    *,
    env: dict[str, TypeExpr],
    signatures: dict[str, ModuleSignature],
    primitive_aliases: dict[str, tuple[str, int]],
    module: AxonModule,
    path: tuple[int, ...],
    dim_symbols: dict[str, DimToken],
    rigid_symbols: set[str],
) -> list[tuple[TypeExpr, ...]]:
    out: list[tuple[TypeExpr, ...]] = []
    for idx, stmt in enumerate(statements):
        stmt_path = (*path, idx)
        if isinstance(stmt, AxonBind):
            inferred = _infer_expr_type(
                stmt.expr,
                env=env,
                signatures=signatures,
                primitive_aliases=primitive_aliases,
                module=module,
                path=(*stmt_path, 0),
                dim_symbols=dim_symbols,
                rigid_symbols=rigid_symbols,
                expected_arity=len(stmt.targets),
            )
            if len(stmt.targets) == 1:
                target = stmt.targets[0]
                if target != "_":
                    if isinstance(inferred, TypeTuple) and len(inferred.items) > 1:
                        raise _error(
                            module,
                            stmt_path,
                            "cannot bind multi-value expression to a single target",
                        )
                    env[target] = inferred
                continue
            if not isinstance(inferred, TypeTuple):
                if isinstance(inferred, TypeList):
                    item_type = inferred.item
                    for target in stmt.targets:
                        if target != "_":
                            env[target] = item_type
                    continue
                raise _error(
                    module,
                    stmt_path,
                    "multi-target bind requires a tuple- or list-valued expression",
                )
            if len(inferred.items) != len(stmt.targets):
                raise _error(
                    module,
                    stmt_path,
                    f"tuple bind arity mismatch: {len(stmt.targets)} target(s), {len(inferred.items)} value(s)",
                )
            for target, item_type in zip(stmt.targets, inferred.items, strict=True):
                if target != "_":
                    env[target] = item_type
            continue
        if isinstance(stmt, AxonReturn):
            ret_types = tuple(
                _infer_expr_type(
                    value,
                    env=env,
                    signatures=signatures,
                    primitive_aliases=primitive_aliases,
                    module=module,
                    path=(*stmt_path, i),
                    dim_symbols=dim_symbols,
                    rigid_symbols=rigid_symbols,
                )
                for i, value in enumerate(stmt.values)
            )
            out.append(ret_types)
            continue
        if isinstance(stmt, AxonRepeat):
            nested_env = dict(env)
            nested_env[stmt.var] = TypeInt()
            out.extend(
                _collect_return_types(
                    stmt.body,
                    env=nested_env,
                    signatures=signatures,
                    primitive_aliases=primitive_aliases,
                    module=module,
                    path=stmt_path,
                    dim_symbols=dim_symbols,
                    rigid_symbols=rigid_symbols,
                )
            )
            for name, tp in nested_env.items():
                if name != stmt.var:
                    env[name] = tp
            continue
        if isinstance(stmt, AxonScopeBind):
            nested_env = dict(env)
            scope_returns = _collect_return_types(
                stmt.body,
                env=nested_env,
                signatures=signatures,
                primitive_aliases=primitive_aliases,
                module=module,
                path=stmt_path,
                dim_symbols=dim_symbols,
                rigid_symbols=rigid_symbols,
            )
            out.extend(scope_returns)
            if scope_returns:
                first = scope_returns[0]
                for target_idx, target in enumerate(stmt.targets):
                    if target == "_":
                        continue
                    if target_idx < len(first):
                        env[target] = first[target_idx]
    return out


def _typecheck_statements(
    statements: tuple[AxonStatement, ...],
    *,
    env: dict[str, TypeExpr],
    signatures: dict[str, ModuleSignature],
    primitive_aliases: dict[str, tuple[str, int]],
    module: AxonModule,
    declared_returns: tuple[TypeExpr, ...],
    path: tuple[int, ...],
    dim_symbols: dict[str, DimToken],
    rigid_symbols: set[str],
) -> None:
    for idx, stmt in enumerate(statements):
        stmt_path = (*path, idx)
        if isinstance(stmt, AxonBind):
            inferred = _infer_expr_type(
                stmt.expr,
                env=env,
                signatures=signatures,
                primitive_aliases=primitive_aliases,
                module=module,
                path=(*stmt_path, 0),
                dim_symbols=dim_symbols,
                rigid_symbols=rigid_symbols,
                expected_arity=len(stmt.targets),
            )
            if len(stmt.targets) == 1:
                target = stmt.targets[0]
                if target != "_":
                    if isinstance(inferred, TypeTuple) and len(inferred.items) > 1:
                        raise _error(
                            module,
                            stmt_path,
                            "cannot bind multi-value expression to a single target",
                        )
                    env[target] = inferred
                continue
            if not isinstance(inferred, TypeTuple):
                if isinstance(inferred, TypeList):
                    item_type = inferred.item
                    for target in stmt.targets:
                        if target != "_":
                            env[target] = item_type
                    continue
                raise _error(
                    module,
                    stmt_path,
                    "multi-target bind requires a tuple- or list-valued expression",
                )
            if len(inferred.items) != len(stmt.targets):
                raise _error(
                    module,
                    stmt_path,
                    f"tuple bind arity mismatch: {len(stmt.targets)} target(s), {len(inferred.items)} value(s)",
                )
            for target, item_type in zip(stmt.targets, inferred.items, strict=True):
                if target != "_":
                    env[target] = item_type
            continue
        if isinstance(stmt, AxonReturn):
            actual_types = tuple(
                _infer_expr_type(
                    value,
                    env=env,
                    signatures=signatures,
                    primitive_aliases=primitive_aliases,
                    module=module,
                    path=(*stmt_path, i),
                    dim_symbols=dim_symbols,
                    rigid_symbols=rigid_symbols,
                )
                for i, value in enumerate(stmt.values)
            )
            if declared_returns:
                if len(actual_types) != len(declared_returns):
                    if len(declared_returns) > 1 and len(actual_types) == 1:
                        packed = actual_types[0]
                        if isinstance(packed, TypeTuple):
                            if len(packed.items) < len(declared_returns):
                                raise _error(
                                    module,
                                    stmt_path,
                                    f"return arity mismatch: expected {len(declared_returns)}, got {len(packed.items)}",
                                )
                            actual_types = tuple(packed.items[: len(declared_returns)])
                        else:
                            raise _error(
                                module,
                                stmt_path,
                                f"return arity mismatch: expected {len(declared_returns)}, got 1",
                            )
                    else:
                        raise _error(
                            module,
                            stmt_path,
                            f"return arity mismatch: expected {len(declared_returns)}, got {len(actual_types)}",
                        )
                dim_subst: dict[str, DimToken] = {}
                for ret_idx, (actual, expected) in enumerate(
                    zip(actual_types, declared_returns, strict=True)
                ):
                    if not _types_compatible(
                        actual,
                        expected,
                        dim_subst=dim_subst,
                        dim_symbols=dim_symbols,
                        rigid_symbols=rigid_symbols,
                    ):
                        raise _error(
                            module,
                            stmt_path,
                            f"return type mismatch at index {ret_idx}: expected {render_type(expected)}, got {render_type(actual)}",
                        )
            continue
        if isinstance(stmt, AxonRepeat):
            to_type = _infer_expr_type(
                stmt.to_expr,
                env=env,
                signatures=signatures,
                primitive_aliases=primitive_aliases,
                module=module,
                path=(*stmt_path, 0),
                dim_symbols=dim_symbols,
                rigid_symbols=rigid_symbols,
            )
            from_type = _infer_expr_type(
                stmt.from_expr,
                env=env,
                signatures=signatures,
                primitive_aliases=primitive_aliases,
                module=module,
                path=(*stmt_path, 1),
                dim_symbols=dim_symbols,
                rigid_symbols=rigid_symbols,
            )
            step_type = _infer_expr_type(
                stmt.step_expr,
                env=env,
                signatures=signatures,
                primitive_aliases=primitive_aliases,
                module=module,
                path=(*stmt_path, 2),
                dim_symbols=dim_symbols,
                rigid_symbols=rigid_symbols,
            )
            if not (is_numeric_type(to_type) or isinstance(to_type, TypeNamed)):
                raise _error(
                    module,
                    stmt_path,
                    f"loop upper bound must be numeric, got {render_type(to_type)}",
                )
            if not (is_numeric_type(from_type) or isinstance(from_type, TypeNamed)):
                raise _error(
                    module,
                    stmt_path,
                    f"loop lower bound must be numeric, got {render_type(from_type)}",
                )
            if not (is_numeric_type(step_type) or isinstance(step_type, TypeNamed)):
                raise _error(
                    module, stmt_path, f"loop step must be numeric, got {render_type(step_type)}"
                )
            nested_env = dict(env)
            nested_env[stmt.var] = TypeInt()
            _typecheck_statements(
                stmt.body,
                env=nested_env,
                signatures=signatures,
                primitive_aliases=primitive_aliases,
                module=module,
                declared_returns=(),
                path=stmt_path,
                dim_symbols=dim_symbols,
                rigid_symbols=rigid_symbols,
            )
            for name, tp in nested_env.items():
                if name != stmt.var:
                    env[name] = tp
            continue
        if isinstance(stmt, AxonScopeBind):
            nested_env = dict(env)
            _typecheck_statements(
                stmt.body,
                env=nested_env,
                signatures=signatures,
                primitive_aliases=primitive_aliases,
                module=module,
                declared_returns=(),
                path=stmt_path,
                dim_symbols=dim_symbols,
                rigid_symbols=rigid_symbols,
            )
            return_types = _collect_return_types(
                stmt.body,
                env=nested_env,
                signatures=signatures,
                primitive_aliases=primitive_aliases,
                module=module,
                path=stmt_path,
                dim_symbols=dim_symbols,
                rigid_symbols=rigid_symbols,
            )
            if not return_types:
                raise _error(module, stmt_path, "scope bind body has no typed return path")
            first = return_types[0]
            if len(first) < len(stmt.targets):
                raise _error(
                    module,
                    stmt_path,
                    f"scope bind expects at least {len(stmt.targets)} returned values, got {len(first)}",
                )
            for target_idx, target in enumerate(stmt.targets):
                if target == "_":
                    continue
                env[target] = first[target_idx]
            continue


def typecheck_axon_program(
    modules: tuple[AxonModule, ...], *, main_module: str | None = None
) -> dict[str, ModuleSignature]:
    if not modules:
        raise ValueError("Axon typecheck failed: program must contain at least one module")

    type_aliases = _build_type_alias_table(modules)
    signatures: dict[str, ModuleSignature] = {
        module.name: _module_signature(module, type_aliases=type_aliases) for module in modules
    }
    primitive_aliases = _extract_primitive_aliases(modules, type_aliases=type_aliases)
    dim_symbols: dict[str, DimToken] = {}
    polymorphic_dim_names: set[str] = set()
    for module in modules:
        dim_symbols.update(_symbol_dim_table(module, type_aliases=type_aliases))
        polymorphic_dim_names.update(_module_signature_dim_names(module, type_aliases=type_aliases))
    for name in polymorphic_dim_names:
        dim_symbols.pop(name, None)
    rigid_symbols = set(dim_symbols.keys())
    if len(signatures) != len(modules):
        raise ValueError("Axon typecheck failed: duplicate module names")

    selected_main = modules[-1].name if main_module is None else main_module
    if selected_main not in signatures:
        raise ValueError(f"Axon typecheck failed: unknown main module {selected_main!r}")

    for module in modules:
        sig = signatures[module.name]
        env: dict[str, TypeExpr] = {}
        for name, tp in zip(sig.param_names, sig.params, strict=True):
            env[name] = tp
        if isinstance(module.symbols, dict):
            for name, value in module.symbols.items():
                if value is None:
                    env[name] = TypeInt()
                elif isinstance(value, bool):
                    env[name] = TypeBool()
                elif isinstance(value, int):
                    env[name] = TypeInt()
                elif isinstance(value, float):
                    env[name] = TypeFloat()
                elif isinstance(value, str):
                    env[name] = TypeString()
                else:
                    env[name] = TypeAny()
        for idx, param in enumerate(module.params):
            if not isinstance(param.default_expr, AxonExpr):
                continue
            default_type = _infer_expr_type(
                param.default_expr,
                env=env,
                signatures=signatures,
                primitive_aliases=primitive_aliases,
                module=module,
                path=(-1, idx),
                dim_symbols=dim_symbols,
                rigid_symbols=rigid_symbols,
            )
            expected_type = sig.params[idx]
            if param.optional and not isinstance(expected_type, TypeOptional):
                expected_type = TypeOptional(inner=expected_type)
            if not _types_compatible(
                default_type,
                expected_type,
                dim_subst={},
                dim_symbols=dim_symbols,
                rigid_symbols=rigid_symbols,
            ):
                raise _error(
                    module,
                    (-1, idx),
                    "default value type mismatch for "
                    f"parameter {param.name!r}: expected {render_type(expected_type)}, "
                    f"got {render_type(default_type)}",
                )
            # Optional params with a definite non-null default behave as non-optional
            # inside the module body.
            param_type = sig.params[idx]
            if (
                param.optional
                and isinstance(param_type, TypeOptional)
                and not isinstance(default_type, TypeNull | TypeOptional | TypeAny)
            ):
                env[param.name] = param_type.inner

        _typecheck_statements(
            module.statements,
            env=env,
            signatures=signatures,
            primitive_aliases=primitive_aliases,
            module=module,
            declared_returns=sig.returns,
            path=(),
            dim_symbols=dim_symbols,
            rigid_symbols=rigid_symbols,
        )

    return signatures


def typecheck_axon_module(module: AxonModule) -> ModuleSignature:
    signatures = typecheck_axon_program((module,), main_module=module.name)
    return signatures[module.name]


__all__ = [
    "ModuleSignature",
    "TYPING_RULES",
    "typecheck_axon_module",
    "typecheck_axon_program",
]
