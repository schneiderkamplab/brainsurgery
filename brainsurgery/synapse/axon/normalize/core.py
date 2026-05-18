from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import cast

from ...ops import get_op_lowering_type_signature
from ..ast import (
    AxonBind,
    AxonCond,
    AxonExpr,
    AxonExprAscribe,
    AxonExprBinary,
    AxonExprBind,
    AxonExprCall,
    AxonExprDo,
    AxonExprIf,
    AxonExprLambda,
    AxonExprList,
    AxonExprName,
    AxonExprNull,
    AxonExprParen,
    AxonExprPath,
    AxonExprPipe,
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
    TypePath,
    TypeAliasDef,
    TypeAny,
    TypeExpr,
    TypeList,
    TypeNamed,
    TypeOptional,
    TypeTensor,
    TypeTuple,
    TypeVar,
)
from ..validate import validate_closed_axon_file, validate_normalized_axon_file


def _primitive_op_name(name: str) -> str | None:
    if not name.startswith("_"):
        return None
    return name[1:]


def _is_zero_arg_primitive_name(name: str) -> bool:
    op_name = _primitive_op_name(name)
    if op_name is None:
        return False
    signature = get_op_lowering_type_signature(op_name)
    if signature is None:
        return False
    args = signature.get("args", ())
    kwargs = signature.get("kwargs", {})
    return not args and not kwargs


def _pragma_occurrences(value: object) -> tuple[object, ...]:
    if isinstance(value, dict) and set(value) == {"__pragma_occurrences__"}:
        occurrences = value["__pragma_occurrences__"]
        if isinstance(occurrences, list | tuple):
            return tuple(occurrences)
        raise ValueError("normalize failed: invalid pragma occurrence list")
    return (value,)


def _normalize_single_pragma_value(name: str, value: object) -> object:
    if name == "padding_side":
        side = str(value).strip().lower()
        if side not in {"left", "right"}:
            raise ValueError("PADDING_SIDE must be 'left' or 'right'")
        return side
    if name == "main":
        if isinstance(value, str) and value:
            return value
        raise ValueError("MAIN must be a non-empty string")
    if name == "tokenizer":
        if isinstance(value, str) and value:
            return value
        if isinstance(value, list | tuple) and len(value) == 2:
            checkpoint, tokenizer = value
            if (
                isinstance(checkpoint, str)
                and checkpoint
                and isinstance(tokenizer, str)
                and tokenizer
                ):
                    return (checkpoint, tokenizer)
        if isinstance(value, list | tuple):
            entries: list[tuple[str, str]] = []
            for item in value:
                if not isinstance(item, list | tuple) or len(item) != 2:
                    break
                checkpoint, tokenizer = item
                if not (
                    isinstance(checkpoint, str)
                    and checkpoint
                    and isinstance(tokenizer, str)
                    and tokenizer
                ):
                    break
                entries.append((checkpoint, tokenizer))
            else:
                if entries:
                    return tuple(entries)
        raise ValueError("TOKENIZER must be a non-empty string or a [checkpoint, tokenizer] pair")
    if name == "checkpoints":
        if isinstance(value, str):
            return (value,)
        if isinstance(value, list | tuple):
            items = tuple(str(item) for item in value)
            if not all(isinstance(item, str) and item for item in items):
                raise ValueError("CHECKPOINTS entries must be strings")
            return items
        raise ValueError("CHECKPOINTS must be a string or a list/tuple of strings")
    return value


def _merge_tokenizer_pragma(prev_value: object | None, pragma_value: object) -> object:
    def _entries(value: object | None) -> list[object]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if (
            isinstance(value, tuple)
            and len(value) == 2
            and all(isinstance(item, str) for item in value)
        ):
            return [value]
        if isinstance(value, tuple):
            return list(value)
        raise ValueError("invalid TOKENIZER pragma state")

    entries = _entries(prev_value)
    new_entries = _entries(pragma_value)
    for new_entry in new_entries:
        if isinstance(new_entry, str):
            for entry in entries:
                if isinstance(entry, str):
                    if entry != new_entry:
                        raise ValueError(
                            "conflicting TOKENIZER pragmas; expected a single consistent global tokenizer"
                        )
                    break
            else:
                entries.insert(0, new_entry)
            continue
        checkpoint, tokenizer = cast(tuple[str, str], new_entry)
        for entry in entries:
            if (
                isinstance(entry, tuple)
                and len(entry) == 2
                and all(isinstance(item, str) for item in entry)
                and entry[0] == checkpoint
            ):
                if entry[1] != tokenizer:
                    raise ValueError(
                        "conflicting TOKENIZER pragmas; expected a single tokenizer per checkpoint"
                    )
                break
        else:
            entries.append(new_entry)

    if len(entries) == 1:
        return entries[0]
    return tuple(entries)


def _normalize_pragmas(pragmas: dict[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for name, raw_value in pragmas.items():
        for occurrence in _pragma_occurrences(raw_value):
            value = _normalize_single_pragma_value(name, occurrence)
            prev_value = normalized.get(name)
            if name == "tokenizer":
                normalized[name] = _merge_tokenizer_pragma(prev_value, value)
                continue
            if (
                name in {"main", "padding_side"}
                and isinstance(prev_value, str)
                and prev_value != value
            ):
                raise ValueError(
                    f"conflicting {name.upper()} pragmas; expected a single consistent value"
                )
            normalized[name] = value
    return normalized


def _is_unresolved_generic_type_name(
    type_expr: TypeExpr,
    *,
    type_aliases: dict[str, TypeAliasDef],
) -> bool:
    return (
        isinstance(type_expr, TypeNamed)
        and not type_expr.args
        and type_expr.name != "Tensor"
        and type_expr.name not in type_aliases
        and "." not in type_expr.name
        and "::" not in type_expr.name
    )


def _normalize_type_expr(
    type_expr: TypeExpr | None,
    *,
    type_aliases: dict[str, TypeAliasDef],
) -> TypeExpr | None:
    if type_expr is None:
        return None
    if isinstance(type_expr, TypeVar | TypeAny):
        return type_expr
    if _is_unresolved_generic_type_name(type_expr, type_aliases=type_aliases):
        assert isinstance(type_expr, TypeNamed)
        return TypeVar(type_expr.name)
    if isinstance(type_expr, TypeOptional):
        inner = _normalize_type_expr(type_expr.inner, type_aliases=type_aliases)
        assert inner is not None
        return TypeOptional(inner)
    if isinstance(type_expr, TypeList):
        item = _normalize_type_expr(type_expr.item, type_aliases=type_aliases)
        assert item is not None
        return TypeList(item)
    if isinstance(type_expr, TypeTuple):
        return TypeTuple(
            tuple(
                normalized
                for item in type_expr.items
                for normalized in (_normalize_type_expr(item, type_aliases=type_aliases),)
                if normalized is not None
            )
        )
    if isinstance(type_expr, TypeTensor | TypeNamed):
        return type_expr
    return type_expr


def _normalize_type_aliases(
    type_aliases: dict[str, TypeAliasDef],
    *,
    known_type_aliases: dict[str, TypeAliasDef] | None = None,
) -> dict[str, TypeAliasDef]:
    if not type_aliases:
        return {}
    alias_scope = known_type_aliases if known_type_aliases is not None else type_aliases
    return {
        name: replace(
            alias,
            value=cast(
                TypeExpr,
                _normalize_type_expr(alias.value, type_aliases=alias_scope),
            ),
        )
        for name, alias in type_aliases.items()
    }


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
            raise ValueError(f"normalize failed: invalid callee path sugar {callee!r}")
        path_args.append(AxonExprPath(absolute=absolute, parts=tuple(suffix.split("."))))
        if next_sep < 0:
            break
        rest = rest[next_sep + 1 :]
    return base, tuple(path_args)


def _split_callable_surface_name(name: str) -> tuple[str, str]:
    indexes = [idx for idx in (name.find("@"), name.find("::")) if idx >= 0]
    if not indexes:
        return name, ""
    idx = min(indexes)
    return name[:idx], name[idx:]


def _leading_path_param_count(module: AxonDefinition) -> int:
    count = 0
    for param in module.params:
        if isinstance(param.type_expr, TypePath):
            count += 1
            continue
        break
    return count


def _expand_call_surface(
    expr: AxonExprCall, *, modules_by_name: dict[str, AxonDefinition]
) -> AxonExprCall:
    base_callee, sugared_path_args = _split_callee_path_sugar(expr.callee)
    explicit_args = [*sugared_path_args, *expr.args]
    module = modules_by_name.get(base_callee)
    if module is None:
        if base_callee == expr.callee:
            return replace(expr, args=tuple(explicit_args), kwargs=dict(expr.kwargs))
        return AxonExprCall(callee=base_callee, args=tuple(explicit_args), kwargs=dict(expr.kwargs))

    path_slot_count = len(module.path_params) + _leading_path_param_count(module)
    if len(sugared_path_args) > path_slot_count:
        raise ValueError(
            f"normalize failed: too many path args in call {expr.callee!r} for module {module.name!r}"
        )

    original_kwargs = dict(expr.kwargs)
    if len(explicit_args) < path_slot_count:
        path_param_names = list(module.path_params)
        path_param_names.extend(
            param.name for param in module.params[: _leading_path_param_count(module)]
        )
        for name in path_param_names[len(explicit_args) : path_slot_count]:
            value = original_kwargs.get(name)
            if not isinstance(value, AxonExpr):
                break
            explicit_args.append(value)
            original_kwargs.pop(name, None)

    if len(explicit_args) < path_slot_count:
        raise ValueError(
            f"normalize failed: missing path args in call {expr.callee!r} for module {module.name!r}"
        )

    covered_param_count = max(0, len(explicit_args) - path_slot_count)
    return AxonExprCall(callee=base_callee, args=tuple(explicit_args), kwargs=original_kwargs)


def _module_bound_names(module: AxonDefinition) -> set[str]:
    bound_names = {param.name for param in module.params}
    bound_names.update(name for name in module.path_params if isinstance(name, str))
    if isinstance(module.path_param, str):
        bound_names.add(module.path_param)
    return bound_names


def _is_zero_arg_definition(module: AxonDefinition) -> bool:
    return (
        not module.is_global_binding
        and
        not module.path_params
        and module.path_param is None
        and not module.params
    )


def _canonical_path_expr(expr: AxonExprPath) -> AxonExprPath:
    parts: list[str] = []
    for part in expr.parts:
        parts.extend(piece for piece in part.split(".") if piece)
    return replace(expr, parts=tuple(parts))


def _pipe_stage_to_call(value: AxonExpr, stage: AxonExpr) -> AxonExpr:
    if isinstance(stage, AxonExprName):
        return AxonExprCall(callee=stage.name, args=(value,), kwargs={})
    if isinstance(stage, AxonExprCall):
        return AxonExprCall(
            callee=stage.callee, args=(value, *stage.args), kwargs=dict(stage.kwargs)
        )
    raise ValueError("normalize failed: pipeline stage must be a name or call")


def _normalize_expr(
    expr: AxonExpr,
    *,
    modules_by_name: dict[str, AxonDefinition],
    value_names: set[str],
    bound_names: set[str],
    type_aliases: dict[str, TypeAliasDef],
) -> AxonExpr:
    if isinstance(expr, AxonExprName):
        base, suffix = _split_callable_surface_name(expr.name)
        module = modules_by_name.get(base)
        is_bound_value = base in bound_names or base in value_names
        if (
            module is not None
            and not is_bound_value
            and suffix
        ):
            return _normalize_expr(
                AxonExprCall(callee=expr.name, args=(), kwargs={}),
                modules_by_name=modules_by_name,
                value_names=value_names,
                bound_names=bound_names,
                type_aliases=type_aliases,
            )
        if (
            module is not None
            and not is_bound_value
            and not suffix
            and _is_zero_arg_definition(module)
        ):
            return AxonExprCall(callee=expr.name, args=(), kwargs={})
        if (
            module is None
            and not is_bound_value
            and not suffix
            and _is_zero_arg_primitive_name(expr.name)
        ):
            return AxonExprCall(callee=expr.name, args=(), kwargs={})
        return expr
    if isinstance(expr, AxonExprPath):
        return _canonical_path_expr(expr)
    if isinstance(expr, AxonExprCall):
        args = tuple(
            _normalize_expr(
                arg,
                modules_by_name=modules_by_name,
                value_names=value_names,
                bound_names=bound_names,
                type_aliases=type_aliases,
            )
            for arg in expr.args
        )
        kwargs = {
            key: _normalize_expr(
                value,
                modules_by_name=modules_by_name,
                value_names=value_names,
                bound_names=bound_names,
                type_aliases=type_aliases,
            )
            if isinstance(value, AxonExpr)
            else value
            for key, value in expr.kwargs.items()
        }
        return _expand_call_surface(
            replace(expr, args=args, kwargs=kwargs), modules_by_name=modules_by_name
        )
    if isinstance(expr, AxonExprPipe):
        current = _normalize_expr(
            expr.value,
            modules_by_name=modules_by_name,
            value_names=value_names,
            bound_names=bound_names,
            type_aliases=type_aliases,
        )
        for stage in expr.stages:
            current = _normalize_expr(
                _pipe_stage_to_call(current, stage),
                modules_by_name=modules_by_name,
                value_names=value_names,
                bound_names=bound_names,
                type_aliases=type_aliases,
            )
        return current
    if isinstance(expr, AxonExprBind):
        nested_bound = set(bound_names)
        nested_bound.add(expr.var)
        return replace(
            expr,
            value=_normalize_expr(
                expr.value,
                modules_by_name=modules_by_name,
                value_names=value_names,
                bound_names=bound_names,
                type_aliases=type_aliases,
            ),
            body=_normalize_expr(
                expr.body,
                modules_by_name=modules_by_name,
                value_names=value_names,
                bound_names=nested_bound,
                type_aliases=type_aliases,
            ),
        )
    if isinstance(expr, AxonExprIf | AxonExprTernary):
        cond = _normalize_expr(
            expr.cond,
            modules_by_name=modules_by_name,
            value_names=value_names,
            bound_names=bound_names,
            type_aliases=type_aliases,
        )
        if (
            isinstance(expr, AxonExprTernary)
            and isinstance(cond, AxonExprParen)
            and isinstance(cond.inner, AxonExprCall)
        ):
            cond = cond.inner
        return replace(
            expr,
            cond=cond,
            true_expr=_normalize_expr(
                expr.true_expr,
                modules_by_name=modules_by_name,
                value_names=value_names,
                bound_names=bound_names,
                type_aliases=type_aliases,
            ),
            false_expr=_normalize_expr(
                expr.false_expr,
                modules_by_name=modules_by_name,
                value_names=value_names,
                bound_names=bound_names,
                type_aliases=type_aliases,
            ),
        )
    if isinstance(expr, AxonExprBinary):
        return replace(
            expr,
            left=_normalize_expr(
                expr.left,
                modules_by_name=modules_by_name,
                value_names=value_names,
                bound_names=bound_names,
                type_aliases=type_aliases,
            ),
            right=_normalize_expr(
                expr.right,
                modules_by_name=modules_by_name,
                value_names=value_names,
                bound_names=bound_names,
                type_aliases=type_aliases,
            ),
        )
    if isinstance(expr, AxonExprLambda):
        nested_bound = set(bound_names)
        nested_bound.add(expr.var)
        return replace(
            expr,
            body=_normalize_expr(
                expr.body,
                modules_by_name=modules_by_name,
                value_names=value_names,
                bound_names=nested_bound,
                type_aliases=type_aliases,
            ),
        )
    if isinstance(expr, AxonExprParen):
        inner = _normalize_expr(
            expr.inner,
            modules_by_name=modules_by_name,
            value_names=value_names,
            bound_names=bound_names,
            type_aliases=type_aliases,
        )
        if isinstance(inner, AxonExprAscribe):
            return inner
        return replace(expr, inner=inner)
    if isinstance(expr, AxonExprAscribe):
        inner = _normalize_expr(
            expr.expr,
            modules_by_name=modules_by_name,
            value_names=value_names,
            bound_names=bound_names,
            type_aliases=type_aliases,
        )
        if isinstance(inner, AxonExprParen):
            inner = inner.inner
        return replace(
            expr,
            expr=inner,
            type_expr=cast(
                TypeExpr,
                _normalize_type_expr(expr.type_expr, type_aliases=type_aliases),
            ),
        )
    if isinstance(expr, AxonExprList | AxonExprTuple):
        return replace(
            expr,
            items=tuple(
                _normalize_expr(
                    item,
                    modules_by_name=modules_by_name,
                    value_names=value_names,
                    bound_names=bound_names,
                    type_aliases=type_aliases,
                )
                for item in expr.items
            ),
        )
    if isinstance(expr, AxonExprDo):
        return replace(
            expr,
            body=_normalize_statements(
                expr.body,
                modules_by_name=modules_by_name,
                value_names=value_names,
                bound_names=set(bound_names),
                type_aliases=type_aliases,
            ),
        )
    return expr


def _normalize_repeat_yield(stmt: AxonRepeat) -> AxonRepeat:
    if stmt.body and isinstance(stmt.body[-1], AxonYield):
        if stmt.carry is not None:
            return stmt
        if stmt.targets is not None:
            return replace(stmt, carry=stmt.targets)
        yielded_names: list[str] = []
        for value in stmt.body[-1].values:
            if not isinstance(value, AxonExprName) or value.name == "_":
                return stmt
            yielded_names.append(value.name)
        if yielded_names:
            targets = tuple(yielded_names)
            return replace(stmt, targets=targets, carry=targets)
        return stmt
    if stmt.targets is None:
        return replace(stmt, body=(*stmt.body, AxonYield(values=(AxonExprNull(),))))
    normalized_body = tuple(
        [
            *stmt.body,
            AxonYield(values=tuple(AxonExprName(name=name) for name in stmt.targets)),
        ]
    )
    return replace(
        stmt,
        body=normalized_body,
        carry=stmt.targets if stmt.carry is None else stmt.carry,
    )


def _normalize_statement(
    stmt: AxonStatement,
    *,
    modules_by_name: dict[str, AxonDefinition],
    value_names: set[str],
    bound_names: set[str],
    type_aliases: dict[str, TypeAliasDef],
) -> AxonStatement:
    if isinstance(stmt, AxonBind):
        return replace(
            stmt,
            expr=_normalize_expr(
                stmt.expr,
                modules_by_name=modules_by_name,
                value_names=value_names,
                bound_names=bound_names,
                type_aliases=type_aliases,
            ),
        )
    if isinstance(stmt, AxonReturn | AxonYield):
        return replace(
            stmt,
            values=tuple(
                _normalize_expr(
                    value,
                    modules_by_name=modules_by_name,
                    value_names=value_names,
                    bound_names=bound_names,
                    type_aliases=type_aliases,
                )
                for value in stmt.values
            ),
        )
    if isinstance(stmt, AxonCond):
        return replace(
            stmt,
            cond=_normalize_expr(
                stmt.cond,
                modules_by_name=modules_by_name,
                value_names=value_names,
                bound_names=bound_names,
                type_aliases=type_aliases,
            ),
            true_body=_normalize_statements(
                stmt.true_body,
                modules_by_name=modules_by_name,
                value_names=value_names,
                bound_names=set(bound_names),
                type_aliases=type_aliases,
            ),
            false_body=_normalize_statements(
                stmt.false_body,
                modules_by_name=modules_by_name,
                value_names=value_names,
                bound_names=set(bound_names),
                type_aliases=type_aliases,
            ),
        )
    if isinstance(stmt, AxonRepeat):
        loop_bound = set(bound_names)
        loop_bound.add(stmt.var)
        if stmt.carry:
            loop_bound.update(name for name in stmt.carry if name != "_")
        normalized = replace(
            stmt,
            from_expr=_normalize_expr(
                stmt.from_expr,
                modules_by_name=modules_by_name,
                value_names=value_names,
                bound_names=bound_names,
                type_aliases=type_aliases,
            ),
            to_expr=_normalize_expr(
                stmt.to_expr,
                modules_by_name=modules_by_name,
                value_names=value_names,
                bound_names=bound_names,
                type_aliases=type_aliases,
            ),
            step_expr=_normalize_expr(
                stmt.step_expr,
                modules_by_name=modules_by_name,
                value_names=value_names,
                bound_names=bound_names,
                type_aliases=type_aliases,
            ),
            body=_normalize_statements(
                stmt.body,
                modules_by_name=modules_by_name,
                value_names=value_names,
                bound_names=loop_bound,
                type_aliases=type_aliases,
            ),
        )
        return _normalize_repeat_yield(normalized)
    if isinstance(stmt, AxonScopeBind):
        return replace(
            stmt,
            prefix=_canonical_path_expr(stmt.prefix),
            kwargs={
                key: _normalize_expr(
                    value,
                    modules_by_name=modules_by_name,
                    value_names=value_names,
                    bound_names=bound_names,
                    type_aliases=type_aliases,
                )
                if isinstance(value, AxonExpr)
                else value
                for key, value in stmt.kwargs.items()
            },
            body=_normalize_statements(
                stmt.body,
                modules_by_name=modules_by_name,
                value_names=value_names,
                bound_names=set(bound_names),
                type_aliases=type_aliases,
            ),
        )
    return stmt


def _normalize_statements(
    statements: tuple[AxonStatement, ...],
    *,
    modules_by_name: dict[str, AxonDefinition],
    value_names: set[str],
    bound_names: set[str],
    type_aliases: dict[str, TypeAliasDef],
) -> tuple[AxonStatement, ...]:
    normalized: list[AxonStatement] = []
    local_bound = set(bound_names)
    for stmt in statements:
        normalized_stmt = _normalize_statement(
            stmt,
            modules_by_name=modules_by_name,
            value_names=value_names,
            bound_names=local_bound,
            type_aliases=type_aliases,
        )
        normalized.append(normalized_stmt)
        if isinstance(normalized_stmt, AxonBind | AxonScopeBind):
            local_bound.update(target for target in normalized_stmt.targets if target != "_")
        elif isinstance(normalized_stmt, AxonRepeat) and normalized_stmt.targets is not None:
            local_bound.update(target for target in normalized_stmt.targets if target != "_")
    return tuple(normalized)


def _normalize_module(
    module: AxonDefinition,
    *,
    modules_by_name: dict[str, AxonDefinition],
    value_names: set[str],
    type_aliases: dict[str, TypeAliasDef],
) -> AxonDefinition:
    bound_names = _module_bound_names(module)
    return replace(
        module,
        type_aliases=_normalize_type_aliases(
            dict(module.type_aliases or {}),
            known_type_aliases=type_aliases,
        )
        or None,
        params=tuple(
            replace(
                param,
                type_expr=_normalize_type_expr(param.type_expr, type_aliases=type_aliases),
                default_expr=_normalize_expr(
                    param.default_expr,
                    modules_by_name=modules_by_name,
                    value_names=value_names,
                    bound_names=bound_names,
                    type_aliases=type_aliases,
                )
                if param.default_expr is not None
                else None,
            )
            for param in module.params
        ),
        statements=_normalize_statements(
            module.statements,
            modules_by_name=modules_by_name,
            value_names=value_names,
            bound_names=bound_names,
            type_aliases=type_aliases,
        ),
        body_expr=(
            _normalize_expr(
                module.body_expr,
                modules_by_name=modules_by_name,
                value_names=value_names,
                bound_names=bound_names,
                type_aliases=type_aliases,
            )
            if module.body_expr is not None
            else None
        ),
        return_type_expr=_normalize_type_expr(module.return_type_expr, type_aliases=type_aliases),
    )


def normalize_closed_axon_file(program: AxonFile, *, main_module: str | None = None) -> AxonFile:
    validate_closed_axon_file(program, main_module=main_module)
    modules_by_name = {module.name: module for module in program.modules}
    type_aliases = dict(program.type_aliases)
    for module in program.modules:
        type_aliases.update(module.type_aliases or {})
    value_names: set[str] = set()
    normalized_modules = tuple(
        _normalize_module(
            module,
            modules_by_name=modules_by_name,
            value_names=value_names,
            type_aliases=type_aliases,
        )
        for module in program.modules
    )
    normalized = replace(
        program,
        modules=normalized_modules,
        pragmas=_normalize_pragmas(dict(program.pragmas)),
        type_aliases=_normalize_type_aliases(
            dict(program.type_aliases),
            known_type_aliases=type_aliases,
        ),
    )
    validate_normalized_axon_file(normalized, main_module=main_module)
    return normalized


__all__ = ["normalize_closed_axon_file"]
