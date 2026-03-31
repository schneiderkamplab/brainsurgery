from __future__ import annotations

import re
from typing import Iterable

from .call_parser import split_top_level
from .types import (
    AxonBind,
    AxonModule,
    AxonRepeat,
    AxonReturn,
    AxonScope,
    AxonScopeBind,
    AxonStatement,
)

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _stmt_path(path: tuple[int, ...]) -> str:
    if not path:
        return "root"
    return "root." + ".".join(str(i) for i in path)


def _error(module: AxonModule, path: tuple[int, ...], message: str) -> ValueError:
    return ValueError(
        f"Axon AST validation failed in module '{module.name}' at {_stmt_path(path)}: {message}"
    )


def _iter_nested(stmt: AxonStatement) -> Iterable[AxonStatement]:
    if isinstance(stmt, AxonRepeat):
        return stmt.body
    if isinstance(stmt, AxonScopeBind):
        return stmt.body
    if isinstance(stmt, AxonScope):
        return stmt.body
    return ()


def _has_compatible_return(stmts: tuple[AxonStatement, ...], min_arity: int) -> bool:
    for stmt in stmts:
        if isinstance(stmt, AxonReturn) and len(stmt.values) >= min_arity:
            return True
        nested = tuple(_iter_nested(stmt))
        if nested and _has_compatible_return(nested, min_arity):
            return True
    return False


def _expected_return_arity(return_type_expr: str | None) -> int | None:
    if not isinstance(return_type_expr, str):
        return None
    text = return_type_expr.strip()
    if not text:
        return None
    if text.startswith("(") and text.endswith(")"):
        inner = text[1:-1].strip()
        if not inner:
            return 0
        return len(split_top_level(inner, ","))
    return 1


def _validate_name(name: str, *, module: AxonModule, path: tuple[int, ...], field: str) -> None:
    if name == "_":
        return
    if _NAME_RE.fullmatch(name) is None:
        raise _error(
            module, path, f"invalid {field} name {name!r}; expected [A-Za-z_][A-Za-z0-9_]*"
        )


def _has_duplicate_non_discard(names: tuple[str, ...]) -> bool:
    non_discard = [name for name in names if name != "_"]
    return len(set(non_discard)) != len(non_discard)


def _validate_statement(stmt: AxonStatement, module: AxonModule, path: tuple[int, ...]) -> None:
    if isinstance(stmt, AxonScope):
        raise _error(
            module,
            path,
            "scope statement form is not supported; use '<target> <- scope@name do ... return ...'",
        )

    if isinstance(stmt, AxonBind):
        if not stmt.targets:
            raise _error(module, path, "binding must contain at least one target")
        if _has_duplicate_non_discard(stmt.targets):
            raise _error(module, path, "binding contains duplicate targets")
        for name in stmt.targets:
            _validate_name(name, module=module, path=path, field="binding target")
        if not stmt.expr.strip():
            raise _error(module, path, "binding expression cannot be empty")
        return

    if isinstance(stmt, AxonReturn):
        if not stmt.values:
            raise _error(module, path, "return must contain at least one value")
        for value in stmt.values:
            if not str(value).strip():
                raise _error(module, path, "return values must be non-empty")
        return

    if isinstance(stmt, AxonRepeat):
        _validate_name(stmt.var, module=module, path=path, field="loop variable")
        if not stmt.to_expr.strip():
            raise _error(module, path, "for-loop upper bound cannot be empty")
        if not stmt.from_expr.strip():
            raise _error(module, path, "for-loop lower bound cannot be empty")
        if not stmt.step_expr.strip():
            raise _error(module, path, "for-loop step cannot be empty")
        if not stmt.body:
            raise _error(module, path, "for-loop body cannot be empty")
        for i, child in enumerate(stmt.body):
            _validate_statement(child, module, (*path, i))
        return

    if isinstance(stmt, AxonScopeBind):
        if not stmt.targets:
            raise _error(module, path, "scope bind must contain at least one target")
        if _has_duplicate_non_discard(stmt.targets):
            raise _error(module, path, "scope bind contains duplicate targets")
        for name in stmt.targets:
            _validate_name(name, module=module, path=path, field="scope bind target")
        if not stmt.prefix.strip():
            raise _error(module, path, "scope bind prefix cannot be empty")
        if not stmt.body:
            raise _error(module, path, "scope bind body cannot be empty")
        for i, child in enumerate(stmt.body):
            _validate_statement(child, module, (*path, i))
        if not _has_compatible_return(stmt.body, len(stmt.targets)):
            raise _error(
                module,
                path,
                f"scope bind requires a reachable return with at least {len(stmt.targets)} value(s)",
            )
        return


def _validate_module(module: AxonModule) -> None:
    if not module.name.strip():
        raise ValueError("Axon AST validation failed: module name cannot be empty")

    param_names = [param.name for param in module.params]
    for name in param_names:
        _validate_name(name, module=module, path=(), field="parameter")
    duplicates = sorted({name for name in param_names if param_names.count(name) > 1})
    if duplicates:
        raise ValueError(
            f"Axon AST validation failed in module '{module.name}': duplicate parameter name(s): {', '.join(duplicates)}"
        )
    if module.path_params:
        overlap = sorted(set(param_names) & set(module.path_params))
        if overlap:
            names = ", ".join(overlap)
            raise ValueError(
                f"Axon AST validation failed in module '{module.name}': path parameter(s) conflict with value parameter(s): {names}"
            )
    expected_arity = _expected_return_arity(module.return_type_expr)
    for i, stmt in enumerate(module.statements):
        _validate_statement(stmt, module, (i,))
        if expected_arity is not None and isinstance(stmt, AxonReturn):
            actual = len(stmt.values)
            if actual != expected_arity:
                # A single return expression may evaluate to a tuple (e.g., point-free alias
                # to a multi-output primitive). Without full type inference on expressions,
                # this case is ambiguous, so avoid rejecting it here.
                if expected_arity > 1 and actual == 1:
                    continue
                raise _error(
                    module,
                    (i,),
                    f"return arity mismatch: signature implies {expected_arity} value(s), got {actual}",
                )


def validate_axon_program(
    modules: tuple[AxonModule, ...], *, main_module: str | None = None
) -> None:
    if not modules:
        raise ValueError("Axon AST validation failed: program must contain at least one module")
    names = [module.name for module in modules]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(
            "Axon AST validation failed: duplicate module name(s): " + ", ".join(duplicates)
        )
    selected_main = modules[-1].name if main_module is None else main_module
    if selected_main not in set(names):
        raise ValueError(f"Axon AST validation failed: unknown main module {selected_main!r}")
    for module in modules:
        _validate_module(module)


__all__ = ["validate_axon_program"]
