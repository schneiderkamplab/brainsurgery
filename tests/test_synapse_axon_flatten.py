from __future__ import annotations

from pathlib import Path

from brainsurgery.synapse.axon.ast import (
    AxonBind,
    AxonCond,
    AxonExpr,
    AxonExprAscribe,
    AxonExprBinary,
    AxonExprBind,
    AxonExprBool,
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
    AxonRepeat,
    AxonReturn,
    AxonScopeBind,
    TypeInt,
    TypeNamed,
    TypeOptional,
    TypePath,
    TypeTensor,
    ast_equal,
    render_axon_file,
)
from brainsurgery.synapse.axon.elaborate import elaborate_closed_axon_file
from brainsurgery.synapse.axon.flatten import flatten_closed_axon_file
from brainsurgery.synapse.axon.normalize import normalize_closed_axon_file
from brainsurgery.synapse.axon.parse import parse_axon_program
from brainsurgery.synapse.axon.resolve import resolve_axon_program_from_path
from brainsurgery.synapse.axon.validate import validate_flat_axon_file


def _flatten(program, *, main_module: str):
    normalized = normalize_closed_axon_file(program, main_module=main_module)
    elaborated = elaborate_closed_axon_file(normalized, main_module=main_module)
    return flatten_closed_axon_file(elaborated, main_module=main_module)


def _walk_expr(expr: AxonExpr):
    yield expr
    if isinstance(expr, AxonExprBinary):
        yield from _walk_expr(expr.left)
        yield from _walk_expr(expr.right)
    elif isinstance(expr, AxonExprBind):
        yield from _walk_expr(expr.value)
        yield from _walk_expr(expr.body)
    elif isinstance(expr, AxonExprCall):
        for arg in expr.args:
            yield from _walk_expr(arg)
        for value in expr.kwargs.values():
            if isinstance(value, AxonExpr):
                yield from _walk_expr(value)
    elif isinstance(expr, AxonExprDo):
        for stmt in expr.body:
            yield from _walk_stmt(stmt)
    elif isinstance(expr, AxonExprIf | AxonExprTernary):
        yield from _walk_expr(expr.cond)
        yield from _walk_expr(expr.true_expr)
        yield from _walk_expr(expr.false_expr)
    elif isinstance(expr, AxonExprLambda):
        yield from _walk_expr(expr.body)
    elif isinstance(expr, AxonExprAscribe):
        yield from _walk_expr(expr.expr)
    elif isinstance(expr, AxonExprList | AxonExprTuple):
        for item in expr.items:
            yield from _walk_expr(item)
    elif isinstance(expr, AxonExprParen):
        yield from _walk_expr(expr.inner)
    elif isinstance(expr, AxonExprPipe):
        yield from _walk_expr(expr.value)
        for stage in expr.stages:
            yield from _walk_expr(stage)


def _walk_stmt(stmt):
    if isinstance(stmt, AxonBind):
        yield from _walk_expr(stmt.expr)
    else:
        values = getattr(stmt, "values", ())
        for value in values:
            yield from _walk_expr(value)
        if isinstance(stmt, AxonCond):
            yield from _walk_expr(stmt.cond)
            for inner in stmt.true_body:
                yield from _walk_stmt(inner)
            for inner in stmt.false_body:
                yield from _walk_stmt(inner)
            return
        for attr in ("from_expr", "to_expr", "step_expr"):
            if hasattr(stmt, attr):
                yield from _walk_expr(getattr(stmt, attr))
        if hasattr(stmt, "kwargs"):
            for value in stmt.kwargs.values():
                if isinstance(value, AxonExpr):
                    yield from _walk_expr(value)
        if hasattr(stmt, "body"):
            for inner in stmt.body:
                yield from _walk_stmt(inner)


def test_flatten_normalizes_body_expr_and_removes_pipe() -> None:
    source = """
g :: Tensor[B,S,D] -> Tensor[B,S,D]
g x = x

h :: Tensor[B,S,D] -> Tensor[B,S,D]
h x = x

main :: Tensor[B,S,D] -> Tensor[B,S,D]
main x = g x |> h
"""
    program = parse_axon_program(source)
    flat = _flatten(program, main_module="main")
    validate_flat_axon_file(flat, main_module="main")
    module = next(module for module in flat.modules if module.name == "main")
    assert module.body_expr is None
    assert len(module.statements) >= 2
    assert not any(
        isinstance(expr, AxonExprPipe) for stmt in module.statements for expr in _walk_stmt(stmt)
    )


def test_flatten_lifts_nested_call_arguments_into_temps() -> None:
    source = """
f :: Tensor[B,S,D] -> Tensor[B,S,D] -> Tensor[B,S,D]
f x y = x

g :: Tensor[B,S,D] -> Tensor[B,S,D]
g x = x

h :: Tensor[B,S,D] -> Tensor[B,S,D]
h x = x

main :: Tensor[B,S,D] -> Tensor[B,S,D]
main x = do
  y <- f (g x) (h x)
  return y
"""
    program = parse_axon_program(source)
    flat = _flatten(program, main_module="main")
    validate_flat_axon_file(flat, main_module="main")
    module = next(module for module in flat.modules if module.name == "main")
    binds = [stmt for stmt in module.statements if isinstance(stmt, AxonBind)]
    assert len(binds) >= 3
    temp_targets = [
        stmt.targets[0] for stmt in binds if stmt.targets and stmt.targets[0].startswith("__flat_")
    ]
    assert len(temp_targets) >= 2


def test_flatten_render_reparse_roundtrip() -> None:
    source = """
f :: Tensor[B,S,D] -> Tensor[B,S,D] -> Tensor[B,S,D]
f x y = x

g :: Tensor[B,S,D] -> Tensor[B,S,D]
g x = x

h :: Tensor[B,S,D] -> Tensor[B,S,D]
h x = x

main :: Tensor[B,S,D] -> Tensor[B,S,D]
main x = do
  y <- f (g x) (h x)
  return y
"""
    program = parse_axon_program(source)
    flat = _flatten(program, main_module="main")
    rendered = render_axon_file(flat)
    reparsed = parse_axon_program(rendered)
    reflattened = _flatten(reparsed, main_module="main")
    assert ast_equal(flat, reflattened)


def test_flatten_desugars_conditional_expression_to_if_statement() -> None:
    source = """
main :: Tensor[B,S,D] -> Tensor[B,S,D]
main x = do
  y <- (x == x) ? x : x
  return y
"""
    program = parse_axon_program(source)
    flat = _flatten(program, main_module="main")
    validate_flat_axon_file(flat, main_module="main")
    module = next(module for module in flat.modules if module.name == "main")
    assert not any(isinstance(stmt, AxonCond) for stmt in module.statements)
    ternaries = [
        expr
        for stmt in module.statements
        for expr in _walk_stmt(stmt)
        if isinstance(expr, AxonExprTernary)
    ]
    assert len(ternaries) == 1
    assert not any(
        isinstance(expr, AxonExprIf | AxonExprDo)
        for stmt in module.statements
        for expr in _walk_stmt(stmt)
    )


def test_flatten_eliminates_scopes_and_absolutizes_paths() -> None:
    source = """
use_path :: Path -> Path
use_path p = p

main :: Tensor[B,S,D] -> Tensor[B,S,D]
main x = do
  y <- scope@outer do
    z <- scope@attn do
      p <- use_path @q_proj
      return x
    return z
  return y
"""
    program = parse_axon_program(source)
    flat = _flatten(program, main_module="main")
    validate_flat_axon_file(flat, main_module="main")
    module = next(module for module in flat.modules if module.name == "main")
    assert not any(isinstance(stmt, AxonScopeBind) for stmt in module.statements)
    path_exprs = [
        expr
        for stmt in module.statements
        for expr in _walk_stmt(stmt)
        if isinstance(expr, AxonExprPath)
    ]
    assert AxonExprPath(absolute=True, parts=("outer", "attn", "q_proj")) in path_exprs


def test_flatten_expands_callee_path_sugar_after_elaboration() -> None:
    source = """
use :: Path -> Tensor[B,S,D] -> ?Bool -> ?Int -> Tensor[B,S,D]
use@path x ?flag=false limit = x

main :: Tensor[B,S,D] -> Tensor[B,S,D]
main x = use@proj x
"""
    program = parse_axon_program(source)
    flat = _flatten(program, main_module="main")
    validate_flat_axon_file(flat, main_module="main")
    module = next(module for module in flat.modules if module.name == "main")
    call_bind = next(stmt for stmt in module.statements if isinstance(stmt, AxonBind))
    call_expr = call_bind.expr
    assert isinstance(call_expr, AxonExprCall)
    assert call_expr.callee == "use"
    assert call_expr.args[0] == AxonExprPath(absolute=True, parts=("proj",))
    assert "flag" in call_expr.kwargs
    assert "limit" in call_expr.kwargs


def test_flatten_tail_recurses_repeat_without_step_helper() -> None:
    source = """
L = 4

step :: Tensor[B,S,D] -> Int -> Tensor[B,S,D]
step x i = x

main :: Tensor[B,S,D] -> Tensor[B,S,D]
main x = do
  x <- for@h i <- [0..L) do
    x <- step x i
  return x
"""
    program = parse_axon_program(source)
    flat = _flatten(program, main_module="main")
    validate_flat_axon_file(flat, main_module="main")
    main_module = next(module for module in flat.modules if module.name == "main")
    assert not any(isinstance(stmt, AxonRepeat) for stmt in main_module.statements)
    recur_bind = next(
        stmt
        for stmt in main_module.statements
        if isinstance(stmt, AxonBind)
        and isinstance(stmt.expr, AxonExprCall)
        and stmt.expr.callee.startswith("main__loop_h_recur")
    )
    recur_call = recur_bind.expr
    assert isinstance(recur_call, AxonExprCall)
    recur_module = next(module for module in flat.modules if module.name == recur_call.callee)
    assert recur_module.name.startswith("main__loop_h_recur")
    assert isinstance(recur_module.params[0].type_expr, TypeInt)
    assert isinstance(recur_module.params[1].type_expr, TypeInt)
    assert isinstance(recur_module.params[2].type_expr, TypeInt)
    assert not any(module.name.startswith("main__loop_h_step") for module in flat.modules)
    assert any(module.name.startswith("main__loop_h_recur_continue") for module in flat.modules)


def test_flatten_threads_loop_scope_into_called_module_paths() -> None:
    resolved = resolve_axon_program_from_path(
        Path("brainsurgery/synapse/models/gpt2/generic-gpt2-kv.axon")
    ).ast
    flat = _flatten(resolved, main_module="gpt2")
    validate_flat_axon_file(flat, main_module="gpt2")
    gpt2_block = next(module for module in flat.modules if module.name == "gpt2_block")
    assert gpt2_block.path_param is None
    assert gpt2_block.path_params == ()
    assert gpt2_block.params
    assert gpt2_block.params[0].name == "__scope"
    assert isinstance(gpt2_block.params[0].type_expr, TypePath)
    rendered = render_axon_file(flat)
    assert "gpt2_block __scope" in rendered
    assert "gpt2_block @@'h.{i}'" in rendered
    assert "@@'{__scope}.attn.c_attn'" in rendered


def test_flatten_threads_explicit_path_param_into_absolute_templates() -> None:
    source = """
use :: Path -> Tensor[B,S] -> ?Path -> Tensor[B,S]
use@path x ?weight_path=@weight = x

wrap :: Path -> Tensor[B,S] -> Tensor[B,S]
wrap@__scope x = do
  y <- use@proj x
  return y
"""
    program = parse_axon_program(source)
    flat = _flatten(program, main_module="wrap")
    validate_flat_axon_file(flat, main_module="wrap")
    rendered = render_axon_file(flat)
    assert "@@'{__scope}.proj'" in rendered
    assert "@@'{__scope}.proj.weight'" in rendered


def test_flatten_preserves_explicit_relative_path_kwarg() -> None:
    source = """
use :: Path -> Tensor[B,S] -> ?Path -> Tensor[B,S]
use@path x ?weight_path=@weight = x

wrap :: Path -> Tensor[B,S] -> Tensor[B,S]
wrap@__scope x = do
  y <- use@proj x weight_path=@experts.gate_up_proj
  return y
"""
    program = parse_axon_program(source)
    flat = _flatten(program, main_module="wrap")
    validate_flat_axon_file(flat, main_module="wrap")
    rendered = render_axon_file(flat)
    assert "@@'{__scope}.proj'" in rendered
    assert "@@'{__scope}.experts.gate_up_proj'" in rendered
    assert "@@'{__scope}.proj.weight'" not in rendered


def test_flatten_absolutizes_relative_defaults_for_synthesized_scope_args() -> None:
    source = """
scale :: Tensor[B,S] -> ?Path -> Tensor[B,S]
scale x ?scale_path=@layer_scalar = x

wrap :: Path -> Tensor[B,S] -> Tensor[B,S]
wrap@__scope x = do
  y <- scale x
  return y
"""
    program = parse_axon_program(source)
    flat = _flatten(program, main_module="wrap")
    validate_flat_axon_file(flat, main_module="wrap")
    rendered = render_axon_file(flat)
    assert "scale x" in rendered
    assert "scale_path=@@'{__scope}.layer_scalar'" in rendered


def test_flatten_preserves_forwarded_path_kwarg_for_synthesized_scope_args() -> None:
    source = """
scale :: Tensor[B,S] -> ?Path -> Tensor[B,S]
scale x ?scale_path=@layer_scalar = x

norm :: Path -> Tensor[B,S] -> ?Path -> Tensor[B,S]
norm@path x ?scale_path=@weight = do
  y <- scale x scale_path=scale_path
  return y
"""
    program = parse_axon_program(source)
    flat = _flatten(program, main_module="norm")
    validate_flat_axon_file(flat, main_module="norm")
    rendered = render_axon_file(flat)
    assert "scale x scale_path=scale_path" in rendered
    assert "scale_path=@@'{path}.layer_scalar'" not in rendered


def test_flatten_eliminates_type_aliases() -> None:
    source = """
type TokenIds[B,S] = Tensor[B,S]
type Cache[B,H,T,DH] = ?List[(Tensor[B,H,T,DH], Tensor[B,H,T,DH])]

main :: TokenIds[B,S] -> Cache[B,H,T,DH] -> TokenIds[B,S]
main input_ids past_kv = do
  x <- (input_ids :: TokenIds[B,S])
  return x
"""
    program = parse_axon_program(source)
    flat = _flatten(program, main_module="main")
    validate_flat_axon_file(flat, main_module="main")
    assert flat.type_aliases == {}
    module = next(module for module in flat.modules if module.name == "main")
    assert module.type_aliases is None
    assert isinstance(module.params[0].type_expr, TypeTensor)
    assert isinstance(module.params[1].type_expr, TypeOptional)
    bind_stmt = next(stmt for stmt in module.statements if isinstance(stmt, AxonBind))
    assert isinstance(bind_stmt.expr, AxonExprAscribe)
    assert isinstance(bind_stmt.expr.type_expr, TypeTensor)
    assert not any(
        isinstance(expr.type_expr, TypeNamed)
        for stmt in module.statements
        for expr in _walk_stmt(stmt)
        if isinstance(expr, AxonExprAscribe)
    )
    assert all(
        not isinstance(stmt, AxonCond) for module in flat.modules for stmt in module.statements
    )
    for module in flat.modules:
        for stmt in module.statements:
            if isinstance(stmt, AxonReturn):
                assert all(
                    isinstance(value, (AxonExprAscribe, AxonExprName, AxonExprTuple))
                    for value in stmt.values
                )


def test_flatten_zero_arg_defs_inline_prelude_temps() -> None:
    source = """
main :: Tensor[B,S,D] -> Tensor[B,S,D]
main x = x

A = _config_int (@@text_config.hidden_size) default=(_config_int (@@hidden_size) default=640)
"""
    program = parse_axon_program(source)
    flat = _flatten(program, main_module="main")
    const_module = next(module for module in flat.modules if module.name == "A")
    assert len(const_module.statements) == 3
    default_bind = const_module.statements[0]
    bind_stmt = const_module.statements[1]
    assert isinstance(default_bind, AxonBind)
    assert isinstance(bind_stmt, AxonBind)
    assert isinstance(default_bind.expr, AxonExprCall)
    const_expr = bind_stmt.expr
    assert isinstance(const_expr, AxonExprCall)
    default_expr = const_expr.kwargs["default"]
    assert default_expr == AxonExprName(default_bind.targets[0])
