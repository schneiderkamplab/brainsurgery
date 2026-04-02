from __future__ import annotations

from typing import Iterable

from .type_system import TypeExpr, TypeOptional, TypeTuple
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
    AxonModule,
    AxonRepeat,
    AxonReturn,
    AxonScopeBind,
    AxonStatement,
)


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
    return ()


def _has_compatible_return(stmts: tuple[AxonStatement, ...], min_arity: int) -> bool:
    for stmt in stmts:
        if isinstance(stmt, AxonReturn) and len(stmt.values) >= min_arity:
            return True
        nested = tuple(_iter_nested(stmt))
        if nested and _has_compatible_return(nested, min_arity):
            return True
    return False


def _is_identifier(value: str) -> bool:
    if not value:
        return False
    if not (value[0].isalpha() or value[0] == "_"):
        return False
    return all(ch.isalnum() or ch == "_" for ch in value[1:])


def _expected_return_arity(return_type_expr: TypeExpr | None) -> int | None:
    if return_type_expr is None:
        return None
    current = return_type_expr
    if isinstance(current, TypeOptional):
        current = current.inner
    if isinstance(current, TypeTuple):
        return len(current.items)
    return 1


def _validate_name(name: str, *, module: AxonModule, path: tuple[int, ...], field: str) -> None:
    if name == "_":
        return
    if not _is_identifier(name):
        raise _error(
            module, path, f"invalid {field} name {name!r}; expected [A-Za-z_][A-Za-z0-9_]*"
        )


def _has_duplicate_non_discard(names: tuple[str, ...]) -> bool:
    non_discard = [name for name in names if name != "_"]
    return len(set(non_discard)) != len(non_discard)


def _expr_non_empty(expr: AxonExpr) -> bool:
    if isinstance(expr, AxonExprName):
        return bool(expr.name.strip())
    if isinstance(expr, AxonExprInt | AxonExprFloat | AxonExprBool | AxonExprNull | AxonExprString):
        return True
    if isinstance(expr, AxonExprList):
        return True
    if isinstance(expr, AxonExprTuple):
        return bool(expr.items)
    if isinstance(expr, AxonExprCall):
        return bool(expr.callee.strip())
    if isinstance(expr, AxonExprPipe):
        return bool(expr.stages)
    if isinstance(expr, AxonExprBind):
        return bool(expr.var.strip())
    if isinstance(expr, AxonExprIf | AxonExprTernary):
        return True
    if isinstance(expr, AxonExprBinary):
        return _expr_non_empty(expr.left) and _expr_non_empty(expr.right)
    if isinstance(expr, AxonExprLambda):
        return bool(expr.var.strip())
    if isinstance(expr, AxonExprParen):
        return _expr_non_empty(expr.inner)
    if isinstance(expr, AxonExprDo):
        return bool(expr.body)
    return False


def _validate_expr(expr: AxonExpr, module: AxonModule, path: tuple[int, ...]) -> None:
    if isinstance(expr, AxonExprName):
        _validate_name(expr.name, module=module, path=path, field="expression name")
        return
    if isinstance(expr, AxonExprInt | AxonExprFloat | AxonExprBool | AxonExprNull):
        return
    if isinstance(expr, AxonExprString):
        if not isinstance(expr.value, str):
            raise _error(module, path, "string literal must be a string")
        return
    if isinstance(expr, AxonExprList):
        for i, item in enumerate(expr.items):
            _validate_expr(item, module, (*path, i))
        return
    if isinstance(expr, AxonExprTuple):
        if not expr.items:
            raise _error(module, path, "tuple expression cannot be empty")
        for i, item in enumerate(expr.items):
            _validate_expr(item, module, (*path, i))
        return
    if isinstance(expr, AxonExprCall):
        if not expr.callee.strip():
            raise _error(module, path, "call expression callee cannot be empty")
        for i, arg in enumerate(expr.args):
            _validate_expr(arg, module, (*path, i))
        for key, value in expr.kwargs.items():
            if not _is_identifier(key):
                raise _error(module, path, f"invalid call kwarg name {key!r}")
            if isinstance(value, AxonExpr):
                _validate_expr(value, module, (*path, len(expr.args)))
                continue
            if isinstance(value, bool | int | float | str) or value is None:
                continue
            if isinstance(value, list) and all(
                type(item) is int or isinstance(item, str) for item in value
            ):
                continue
            raise _error(module, path, f"unsupported call kwarg value type for {key!r}")
        return
    if isinstance(expr, AxonExprPipe):
        if not expr.stages:
            raise _error(module, path, "pipe expression must contain at least one stage")
        _validate_expr(expr.value, module, (*path, 0))
        for i, stage in enumerate(expr.stages, start=1):
            _validate_expr(stage, module, (*path, i))
        return
    if isinstance(expr, AxonExprBind):
        _validate_name(expr.var, module=module, path=path, field="bind variable")
        _validate_expr(expr.value, module, (*path, 0))
        _validate_expr(expr.body, module, (*path, 1))
        return
    if isinstance(expr, AxonExprIf):
        _validate_expr(expr.cond, module, (*path, 0))
        _validate_expr(expr.true_expr, module, (*path, 1))
        _validate_expr(expr.false_expr, module, (*path, 2))
        return
    if isinstance(expr, AxonExprTernary):
        _validate_expr(expr.cond, module, (*path, 0))
        _validate_expr(expr.true_expr, module, (*path, 1))
        _validate_expr(expr.false_expr, module, (*path, 2))
        return
    if isinstance(expr, AxonExprBinary):
        if expr.op not in {
            "+",
            "-",
            "*",
            "/",
            "%",
            "==",
            "!=",
            "<",
            "<=",
            ">",
            ">=",
            "and",
            "or",
        }:
            raise _error(module, path, f"unsupported binary operator {expr.op!r}")
        _validate_expr(expr.left, module, (*path, 0))
        _validate_expr(expr.right, module, (*path, 1))
        return
    if isinstance(expr, AxonExprLambda):
        _validate_name(expr.var, module=module, path=path, field="lambda variable")
        _validate_expr(expr.body, module, (*path, 0))
        return
    if isinstance(expr, AxonExprParen):
        _validate_expr(expr.inner, module, (*path, 0))
        return
    if isinstance(expr, AxonExprDo):
        if not expr.body:
            raise _error(module, path, "do expression body cannot be empty")
        for i, stmt in enumerate(expr.body):
            _validate_statement(stmt, module, (*path, i))
        if not _has_compatible_return(expr.body, 1):
            raise _error(
                module, path, "do expression requires a reachable return with at least 1 value"
            )
        return


def _validate_statement(stmt: AxonStatement, module: AxonModule, path: tuple[int, ...]) -> None:
    if isinstance(stmt, AxonBind):
        if not stmt.targets:
            raise _error(module, path, "binding must contain at least one target")
        if _has_duplicate_non_discard(stmt.targets):
            raise _error(module, path, "binding contains duplicate targets")
        for name in stmt.targets:
            _validate_name(name, module=module, path=path, field="binding target")
        _validate_expr(stmt.expr, module, (*path, 0))
        if not _expr_non_empty(stmt.expr):
            raise _error(module, path, "binding expression cannot be empty")
        return

    if isinstance(stmt, AxonReturn):
        if not stmt.values:
            raise _error(module, path, "return must contain at least one value")
        for i, value in enumerate(stmt.values):
            _validate_expr(value, module, (*path, i))
            if not _expr_non_empty(value):
                raise _error(module, path, "return values must be non-empty")
        return

    if isinstance(stmt, AxonRepeat):
        _validate_name(stmt.var, module=module, path=path, field="loop variable")
        _validate_expr(stmt.to_expr, module, (*path, 0))
        _validate_expr(stmt.from_expr, module, (*path, 1))
        _validate_expr(stmt.step_expr, module, (*path, 2))
        if not _expr_non_empty(stmt.to_expr):
            raise _error(module, path, "for-loop upper bound cannot be empty")
        if not _expr_non_empty(stmt.from_expr):
            raise _error(module, path, "for-loop lower bound cannot be empty")
        if not _expr_non_empty(stmt.step_expr):
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
        unsupported = sorted(set(stmt.kwargs) - {"root"})
        if unsupported:
            raise _error(
                module,
                path,
                "scope bind supports only `root` kwarg, got: " + ", ".join(unsupported),
            )
        if "root" in stmt.kwargs:
            root_value = stmt.kwargs["root"]
            if isinstance(root_value, AxonExpr):
                _validate_expr(root_value, module, (*path, len(stmt.targets)))
            elif isinstance(root_value, bool | int | float | str) or root_value is None:
                pass
            elif isinstance(root_value, list) and all(
                type(item) is int or isinstance(item, str) for item in root_value
            ):
                pass
            else:
                raise _error(module, path, "scope bind root must be scalar, list, or expression")
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
    if not _has_compatible_return(module.statements, 0):
        raise ValueError(
            f"Axon AST validation failed in module '{module.name}': module body must contain at least one return statement"
        )
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
