from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace

from ..ast import (
    Constraint,
    ConstraintAtom,
    ConstraintOperand,
    DimExprBinary,
    DimToken,
    TypeDim,
    TypeExpr,
    TypeList,
    TypeNamed,
    TypeOptional,
    TypeTensor,
    TypeTuple,
)
from .core import (
    GraphExpr,
    GraphLiteral,
    GraphModule,
    GraphNode,
    GraphOperand,
    GraphOp,
    GraphPath,
    GraphValue,
    GraphValueRef,
    graph_path_template_names,
)


class UnsupportedConstraintSubstitution(ValueError):
    pass


def substitute_dim_token(
    dim: DimToken,
    subst: Mapping[str, DimToken],
    _cache: dict[int, DimToken] | None = None,
) -> DimToken:
    cache_key = id(dim)
    if _cache is not None and cache_key in _cache:
        return _cache[cache_key]
    if isinstance(dim, str):
        return subst.get(dim, dim)
    if isinstance(dim, DimExprBinary):
        left = substitute_dim_token(dim.left, subst, _cache)
        right = substitute_dim_token(dim.right, subst, _cache)
        if type(left) is int and type(right) is int:
            if dim.op == "+":
                result: DimToken = left + right
                if _cache is not None:
                    _cache[cache_key] = result
                return result
            if dim.op == "-":
                result = left - right
                if _cache is not None:
                    _cache[cache_key] = result
                return result
            if dim.op == "*":
                result = left * right
                if _cache is not None:
                    _cache[cache_key] = result
                return result
            if dim.op == "/" and right != 0 and left % right == 0:
                result = left // right
                if _cache is not None:
                    _cache[cache_key] = result
                return result
        if dim.op == "+":
            if right == 0:
                result = left
                if _cache is not None:
                    _cache[cache_key] = result
                return result
            if left == 0:
                result = right
                if _cache is not None:
                    _cache[cache_key] = result
                return result
        if dim.op == "-":
            if right == 0:
                result = left
                if _cache is not None:
                    _cache[cache_key] = result
                return result
            if left == right:
                result = 0
                if _cache is not None:
                    _cache[cache_key] = result
                return result
            if isinstance(left, DimExprBinary) and left.op == "+":
                if left.left == right:
                    result = left.right
                    if _cache is not None:
                        _cache[cache_key] = result
                    return result
                if left.right == right:
                    result = left.left
                    if _cache is not None:
                        _cache[cache_key] = result
                    return result
            if isinstance(left, DimExprBinary) and left.op == "*":
                if left.left == right and isinstance(left.right, int):
                    remaining = left.right - 1
                    if remaining == 0:
                        result = 0
                        if _cache is not None:
                            _cache[cache_key] = result
                        return result
                    if remaining == 1:
                        result = right
                        if _cache is not None:
                            _cache[cache_key] = result
                        return result
                    result = DimExprBinary(op="*", left=remaining, right=right)
                    if _cache is not None:
                        _cache[cache_key] = result
                    return result
                if left.right == right and isinstance(left.left, int):
                    remaining = left.left - 1
                    if remaining == 0:
                        result = 0
                        if _cache is not None:
                            _cache[cache_key] = result
                        return result
                    if remaining == 1:
                        result = right
                        if _cache is not None:
                            _cache[cache_key] = result
                        return result
                    result = DimExprBinary(op="*", left=remaining, right=right)
                    if _cache is not None:
                        _cache[cache_key] = result
                    return result
        if dim.op == "*":
            if right == 1:
                result = left
                if _cache is not None:
                    _cache[cache_key] = result
                return result
            if left == 1:
                result = right
                if _cache is not None:
                    _cache[cache_key] = result
                return result
            if right == 0 or left == 0:
                result = 0
                if _cache is not None:
                    _cache[cache_key] = result
                return result
            if isinstance(right, DimExprBinary) and right.op == "/" and right.right == left:
                result = right.left
                if _cache is not None:
                    _cache[cache_key] = result
                return result
            if isinstance(left, DimExprBinary) and left.op == "/" and left.right == right:
                result = left.left
                if _cache is not None:
                    _cache[cache_key] = result
                return result
        if dim.op == "/":
            if right == 1:
                result = left
                if _cache is not None:
                    _cache[cache_key] = result
                return result
            if isinstance(left, DimExprBinary) and left.op == "*" and left.right == right:
                result = left.left
                if _cache is not None:
                    _cache[cache_key] = result
                return result
            if isinstance(left, DimExprBinary) and left.op == "*" and left.left == right:
                result = left.right
                if _cache is not None:
                    _cache[cache_key] = result
                return result
        result = DimExprBinary(op=dim.op, left=left, right=right)
        if _cache is not None:
            _cache[cache_key] = result
        return result
    return dim


def substitute_type_expr(
    tp: TypeExpr,
    subst: Mapping[str, DimToken],
    _dim_cache: dict[int, DimToken] | None = None,
    _type_cache: dict[int, TypeExpr] | None = None,
) -> TypeExpr:
    cache_key = id(tp)
    if _type_cache is not None and cache_key in _type_cache:
        return _type_cache[cache_key]
    if isinstance(tp, TypeNamed):
        result: TypeExpr = TypeNamed(
            name=tp.name,
            args=tuple(substitute_dim_token(dim, subst, _dim_cache) for dim in tp.args),
        )
    elif isinstance(tp, TypeOptional):
        result = TypeOptional(substitute_type_expr(tp.inner, subst, _dim_cache, _type_cache))
    elif isinstance(tp, TypeTensor):
        result = TypeTensor(
            base=tp.base,
            dims=tuple(substitute_dim_token(dim, subst, _dim_cache) for dim in tp.dims),
        )
    elif isinstance(tp, TypeList):
        result = TypeList(substitute_type_expr(tp.item, subst, _dim_cache, _type_cache))
    elif isinstance(tp, TypeTuple):
        result = TypeTuple(
            tuple(substitute_type_expr(item, subst, _dim_cache, _type_cache) for item in tp.items)
        )
    else:
        result = tp
    if _type_cache is not None:
        _type_cache[cache_key] = result
    return result


def substitute_constraint_atom(
    atom: ConstraintAtom,
    subst: Mapping[str, DimToken],
) -> ConstraintAtom:
    if isinstance(atom, str):
        replacement = subst.get(atom)
        return replacement if isinstance(replacement, int | str | DimExprBinary) else atom
    if isinstance(atom, DimExprBinary):
        return substitute_dim_token(atom, subst)
    return atom


def substitute_constraint_operand(
    operand: ConstraintOperand,
    subst: Mapping[str, DimToken],
) -> ConstraintOperand:
    if isinstance(operand, tuple):
        return tuple(substitute_constraint_atom(item, subst) for item in operand)
    return substitute_constraint_atom(operand, subst)


def substitute_constraint(constraint: Constraint, subst: Mapping[str, DimToken]) -> Constraint:
    return Constraint(
        relation=constraint.relation,
        left=substitute_constraint_operand(constraint.left, subst),
        right=(
            None
            if constraint.right is None
            else substitute_constraint_operand(constraint.right, subst)
        ),
        guards=tuple(substitute_constraint(guard, subst) for guard in constraint.guards),
    )


def constraint_atom_from_operand(operand: GraphOperand) -> ConstraintAtom:
    if isinstance(operand, GraphValueRef):
        return operand.name
    if isinstance(operand, GraphLiteral):
        if isinstance(operand.value, bool) or operand.value is None or type(operand.value) is int:
            return operand.value
        raise UnsupportedConstraintSubstitution(
            f"cannot represent literal {operand.value!r} in graph constraint"
        )
    raise UnsupportedConstraintSubstitution(
        f"cannot represent {type(operand).__name__} in graph constraint"
    )


def replace_constraint_atom_refs(
    atom: ConstraintAtom,
    subst: Mapping[str, GraphOperand],
) -> ConstraintAtom:
    if isinstance(atom, str):
        replacement = subst.get(atom)
        if replacement is None:
            return atom
        return constraint_atom_from_operand(replacement)
    if isinstance(atom, DimExprBinary):
        left = replace_constraint_atom_refs(atom.left, subst)
        right = replace_constraint_atom_refs(atom.right, subst)
        if (
            isinstance(left, bool)
            or left is None
            or isinstance(right, bool)
            or right is None
        ):
            raise UnsupportedConstraintSubstitution(
                f"cannot substitute non-dim atom into dimension expression {atom!r}"
            )
        if type(left) is int and type(right) is int:
            if atom.op == "+":
                return left + right
            if atom.op == "-":
                return left - right
            if atom.op == "*":
                return left * right
            if atom.op == "/" and right != 0 and left % right == 0:
                return left // right
        return DimExprBinary(op=atom.op, left=left, right=right)
    return atom


def replace_constraint_operand_refs(
    operand: ConstraintOperand,
    subst: Mapping[str, GraphOperand],
) -> ConstraintOperand:
    if isinstance(operand, tuple):
        return tuple(replace_constraint_atom_refs(item, subst) for item in operand)
    return replace_constraint_atom_refs(operand, subst)


def replace_constraint_refs(
    constraint: Constraint,
    subst: Mapping[str, GraphOperand],
) -> Constraint:
    return Constraint(
        relation=constraint.relation,
        left=replace_constraint_operand_refs(constraint.left, subst),
        right=(
            None
            if constraint.right is None
            else replace_constraint_operand_refs(constraint.right, subst)
        ),
        guards=tuple(replace_constraint_refs(guard, subst) for guard in constraint.guards),
    )


def substitute_graph_operand_dims(
    operand: GraphOperand,
    subst: Mapping[str, DimToken],
    _dim_cache: dict[int, DimToken] | None = None,
    _type_cache: dict[int, TypeExpr] | None = None,
    _operand_cache: dict[int, GraphOperand] | None = None,
) -> GraphOperand:
    cache_key = id(operand)
    if _operand_cache is not None and cache_key in _operand_cache:
        return _operand_cache[cache_key]
    if isinstance(operand, GraphValueRef):
        if isinstance(operand.type_expr, TypeDim):
            replacement = subst.get(operand.name)
            if type(replacement) is int:
                return GraphLiteral(value=replacement, type_expr=TypeDim())
            if isinstance(replacement, str):
                return replace(
                    operand,
                    name=replacement,
                    type_expr=substitute_type_expr(operand.type_expr, subst, _dim_cache, _type_cache),
                    dims=(
                        None
                        if operand.dims is None
                        else tuple(substitute_dim_token(dim, subst, _dim_cache) for dim in operand.dims)
                    ),
                )
            if isinstance(replacement, DimExprBinary):
                return _resolved_dim_token_operand(replacement)
        result: GraphOperand = replace(
            operand,
            type_expr=substitute_type_expr(operand.type_expr, subst, _dim_cache, _type_cache),
            dims=(
                None
                if operand.dims is None
                else tuple(substitute_dim_token(dim, subst, _dim_cache) for dim in operand.dims)
            ),
        )
        if _operand_cache is not None:
            _operand_cache[cache_key] = result
        return result
    if isinstance(operand, GraphExpr):
        result = replace(
            operand,
            inputs=tuple(
                substitute_graph_operand_dims(item, subst, _dim_cache, _type_cache, _operand_cache)
                for item in operand.inputs
            ),
            attrs={
                key: substitute_graph_operand_dims(
                    value, subst, _dim_cache, _type_cache, _operand_cache
                )
                for key, value in operand.attrs.items()
            },
            type_expr=substitute_type_expr(operand.type_expr, subst, _dim_cache, _type_cache),
            dims=(
                None
                if operand.dims is None
                else tuple(substitute_dim_token(dim, subst, _dim_cache) for dim in operand.dims)
            ),
        )
        if _operand_cache is not None:
            _operand_cache[cache_key] = result
        return result
    type_expr = getattr(operand, "type_expr", None)
    if type_expr is not None:
        result = replace(
            operand,
            type_expr=substitute_type_expr(type_expr, subst, _dim_cache, _type_cache),
        )
        if _operand_cache is not None:
            _operand_cache[cache_key] = result
        return result
    return operand


def _resolved_dim_token_operand(dim: DimToken) -> GraphOperand:
    """Convert an already-substituted dimension token to a graph operand.

    Dimension substitution is a hygienic parallel substitution. If a callee
    formal `S` is instantiated with the caller expression `S + S`, the `S`
    occurrences inside the replacement belong to the caller scope and must not
    be substituted again.
    """
    if type(dim) is int:
        return GraphLiteral(value=dim, type_expr=TypeDim())
    if isinstance(dim, str):
        return GraphValueRef(name=dim, type_expr=TypeDim())
    if isinstance(dim, DimExprBinary):
        return GraphExpr(
            op=GraphOp(f"core.binary.{dim.op}"),
            inputs=(
                _resolved_dim_token_operand(dim.left),
                _resolved_dim_token_operand(dim.right),
            ),
            attrs={},
            type_expr=TypeDim(),
        )
    return GraphValueRef(name=str(dim), type_expr=TypeDim())


def _dim_token_operand(dim: DimToken, subst: Mapping[str, DimToken]) -> GraphOperand:
    resolved = substitute_dim_token(dim, subst)
    if type(resolved) is int:
        return GraphLiteral(value=resolved, type_expr=TypeDim())
    if isinstance(resolved, str):
        return GraphValueRef(name=resolved, type_expr=TypeDim())
    if isinstance(resolved, DimExprBinary):
        return _resolved_dim_token_operand(resolved)
    return GraphValueRef(name=str(resolved), type_expr=TypeDim())


def substitute_graph_value_dims(
    value: GraphValue,
    subst: Mapping[str, DimToken],
    _dim_cache: dict[int, DimToken] | None = None,
    _type_cache: dict[int, TypeExpr] | None = None,
    _value_cache: dict[int, GraphValue] | None = None,
) -> GraphValue:
    cache_key = id(value)
    if _value_cache is not None and cache_key in _value_cache:
        return _value_cache[cache_key]
    result = replace(
        value,
        type_expr=substitute_type_expr(value.type_expr, subst, _dim_cache, _type_cache),
        dims=(
            None
            if value.dims is None
            else tuple(substitute_dim_token(dim, subst, _dim_cache) for dim in value.dims)
        ),
    )
    if _value_cache is not None:
        _value_cache[cache_key] = result
    return result


def substitute_graph_node_dims(
    node: GraphNode,
    subst: Mapping[str, DimToken],
    _dim_cache: dict[int, DimToken] | None = None,
    _type_cache: dict[int, TypeExpr] | None = None,
    _operand_cache: dict[int, GraphOperand] | None = None,
    _value_cache: dict[int, GraphValue] | None = None,
    _node_cache: dict[int, GraphNode] | None = None,
) -> GraphNode:
    cache_key = id(node)
    if _node_cache is not None and cache_key in _node_cache:
        return _node_cache[cache_key]
    result = replace(
        node,
        inputs=tuple(
            substitute_graph_operand_dims(item, subst, _dim_cache, _type_cache, _operand_cache)
            for item in node.inputs
        ),
        attrs={
            key: substitute_graph_operand_dims(value, subst, _dim_cache, _type_cache, _operand_cache)
            for key, value in node.attrs.items()
        },
        outputs=tuple(
            substitute_graph_value_dims(output, subst, _dim_cache, _type_cache, _value_cache)
            for output in node.outputs
        ),
        type_expr=substitute_type_expr(node.type_expr, subst, _dim_cache, _type_cache),
        dims=(
            None
            if node.dims is None
            else tuple(substitute_dim_token(dim, subst, _dim_cache) for dim in node.dims)
        ),
    )
    if _node_cache is not None:
        _node_cache[cache_key] = result
    return result


def substitute_graph_module_dims(
    module: GraphModule,
    subst: Mapping[str, DimToken],
) -> GraphModule:
    dim_cache: dict[int, DimToken] = {}
    type_cache: dict[int, TypeExpr] = {}
    operand_cache: dict[int, GraphOperand] = {}
    value_cache: dict[int, GraphValue] = {}
    node_cache: dict[int, GraphNode] = {}
    return replace(
        module,
        inputs=tuple(
            substitute_graph_value_dims(value, subst, dim_cache, type_cache, value_cache)
            for value in module.inputs
        ),
        outputs=tuple(
            substitute_graph_operand_dims(output, subst, dim_cache, type_cache, operand_cache)
            for output in module.outputs
        ),
        nodes=tuple(
            substitute_graph_node_dims(
                node,
                subst,
                dim_cache,
                type_cache,
                operand_cache,
                value_cache,
                node_cache,
            )
            for node in module.nodes
        ),
        return_type_expr=(
            None
            if module.return_type_expr is None
            else substitute_type_expr(module.return_type_expr, subst, dim_cache, type_cache)
        ),
        constraints=tuple(substitute_constraint(item, subst) for item in module.constraints),
    )


def operand_path_fragment(operand: GraphOperand) -> tuple[bool, tuple[str, ...]] | None:
    from .core import GraphLiteral

    if isinstance(operand, GraphPath):
        return operand.absolute, operand.parts
    if isinstance(operand, GraphValueRef):
        return False, (f"{{{operand.name}}}",)
    if isinstance(operand, GraphLiteral) and isinstance(operand.value, str | int):
        return False, (str(operand.value),)
    return None


def replace_path_template_refs(path: GraphPath, subst: Mapping[str, GraphOperand]) -> GraphPath:
    absolute = path.absolute
    parts: list[str] = []
    changed = False
    for part in path.parts:
        names = graph_path_template_names(GraphPath(absolute=path.absolute, parts=(part,)))
        if not names:
            parts.append(part)
            continue
        if part.startswith("{") and part.endswith("}") and part[1:-1] in subst:
            replacement = operand_path_fragment(subst[part[1:-1]])
            if replacement is not None:
                repl_absolute, repl_parts = replacement
                absolute = absolute or repl_absolute
                parts.extend(repl_parts)
                changed = True
                continue
        rewritten = part
        for name in sorted(names, key=len, reverse=True):
            if name not in subst:
                continue
            replacement = operand_path_fragment(subst[name])
            if replacement is None:
                continue
            repl_absolute, repl_parts = replacement
            absolute = absolute or repl_absolute
            rewritten = rewritten.replace("{" + name + "}", ".".join(repl_parts))
            changed = True
        parts.append(rewritten)
    if not changed:
        return path
    return GraphPath(absolute=absolute, parts=tuple(part for part in parts if part))


def replace_operand_refs(
    operand: GraphOperand,
    subst: Mapping[str, GraphOperand],
    *,
    fold_operand: Callable[[GraphOperand], GraphOperand] | None = None,
) -> GraphOperand:
    if isinstance(operand, GraphValueRef):
        return subst.get(operand.name, operand)
    if isinstance(operand, GraphPath):
        return replace_path_template_refs(operand, subst)
    if isinstance(operand, GraphExpr):
        rewritten = replace(
            operand,
            inputs=tuple(
                replace_operand_refs(item, subst, fold_operand=fold_operand)
                for item in operand.inputs
            ),
            attrs={
                key: replace_operand_refs(value, subst, fold_operand=fold_operand)
                for key, value in operand.attrs.items()
            },
        )
        return fold_operand(rewritten) if fold_operand is not None else rewritten
    return operand


def rename_operand(operand: GraphOperand, renames: Mapping[str, str]) -> GraphOperand:
    if isinstance(operand, GraphValueRef):
        return replace(operand, name=renames.get(operand.name, operand.name))
    if isinstance(operand, GraphPath):
        parts: list[str] = []
        changed = False
        for part in operand.parts:
            rewritten = part
            for old, new in sorted(renames.items(), key=lambda item: len(item[0]), reverse=True):
                if old == new:
                    continue
                before = rewritten
                rewritten = rewritten.replace("{" + old + "}", "{" + new + "}")
                changed = changed or rewritten != before
            parts.append(rewritten)
        if not changed:
            return operand
        return GraphPath(absolute=operand.absolute, parts=tuple(parts))
    if isinstance(operand, GraphExpr):
        return replace(
            operand,
            inputs=tuple(rename_operand(item, renames) for item in operand.inputs),
            attrs={key: rename_operand(value, renames) for key, value in operand.attrs.items()},
        )
    return operand


__all__ = [
    "UnsupportedConstraintSubstitution",
    "constraint_atom_from_operand",
    "operand_path_fragment",
    "replace_constraint_atom_refs",
    "replace_constraint_operand_refs",
    "replace_constraint_refs",
    "rename_operand",
    "replace_operand_refs",
    "replace_path_template_refs",
    "substitute_constraint",
    "substitute_constraint_atom",
    "substitute_constraint_operand",
    "substitute_dim_token",
    "substitute_graph_module_dims",
    "substitute_graph_node_dims",
    "substitute_graph_operand_dims",
    "substitute_graph_value_dims",
    "substitute_type_expr",
]
