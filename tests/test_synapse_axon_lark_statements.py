from __future__ import annotations

import pytest

from brainsurgery.synapse.axon.ast import (
    AxonBind,
    AxonExprCall,
    AxonExprDo,
    AxonExprInt,
    AxonExprList,
    AxonExprPath,
    AxonExprString,
    AxonRepeat,
    AxonReturn,
    AxonScopeBind,
    AxonYield,
)
from brainsurgery.synapse.axon.parse import parse_axon_program


def _parse_rhs_do(source: str) -> AxonExprDo:
    parsed = parse_axon_program(source)
    assert len(parsed.modules) == 1
    rhs = parsed.modules[0].body_expr
    assert isinstance(rhs, AxonExprDo)
    return rhs


def test_parse_for_statement_with_step() -> None:
    source = """
lin :: Tensor[B,S,D] -> Tensor[B,S,D]
lin x = do
  for@layers i <- [1..8) step=2 i
  return x
"""
    rhs = _parse_rhs_do(source)
    assert len(rhs.body) == 2
    stmt = rhs.body[0]
    assert isinstance(stmt, AxonRepeat)
    assert stmt.name == "layers"
    assert stmt.var == "i"
    assert stmt.from_expr == AxonExprInt(value=1)
    assert stmt.to_expr == AxonExprInt(value=8)
    assert stmt.step_expr == AxonExprInt(value=2)


def test_parse_for_bind_with_carry_and_yield() -> None:
    source = """
lin :: Tensor[B,S,D] -> Tensor[B,S,D]
lin x = do
  y <- for@layers i <- [0..2) carry (x) do
    x <- lin x
    yield x
  return y
"""
    rhs = _parse_rhs_do(source)
    stmt = rhs.body[0]
    assert isinstance(stmt, AxonRepeat)
    assert stmt.targets == ("y",)
    assert stmt.carry == ("x",)
    assert isinstance(stmt.body[-1], AxonYield)
    assert len(stmt.body[-1].values) == 1


def test_parse_for_bind_short_form_without_carry_or_yield() -> None:
    source = """
lin :: Tensor[B,S,D] -> Tensor[B,S,D]
lin x = do
  y <- for@layers i <- [0..2) do
    y <- lin x
  return y
"""
    rhs = _parse_rhs_do(source)
    stmt = rhs.body[0]
    assert isinstance(stmt, AxonRepeat)
    assert stmt.targets == ("y",)
    assert stmt.carry is None


def test_parse_scope_bind_statement_without_at() -> None:
    source = """
lin :: Tensor[B,S,D] -> Tensor[B,S,D]
lin x = do
  y, z <- scope model.layers do return x, x
  return y
"""
    rhs = _parse_rhs_do(source)
    stmt = rhs.body[0]
    assert isinstance(stmt, AxonScopeBind)
    assert stmt.targets == ("y", "z")
    assert stmt.prefix == AxonExprPath(absolute=False, parts=("model", "layers"))


def test_parse_scope_bind_statement_with_root_kwarg() -> None:
    source = """
lin :: Tensor[B,S,D] -> Tensor[B,S,D]
lin x = do
  y <- scope model.layers root=["model", "language_model.model"] do return x
  return y
"""
    rhs = _parse_rhs_do(source)
    stmt = rhs.body[0]
    assert isinstance(stmt, AxonScopeBind)
    assert stmt.prefix == AxonExprPath(absolute=False, parts=("model", "layers"))
    root = stmt.kwargs.get("root")
    assert isinstance(root, AxonExprList)
    assert root.items == (
        AxonExprString(value="model"),
        AxonExprString(value="language_model.model"),
    )


def test_parse_scope_statement_is_rejected() -> None:
    source = """
lin :: Tensor[B,S,D] -> Tensor[B,S,D]
lin x = do
  scope@attn do
    return x
"""
    with pytest.raises(ValueError, match="invalid Axon source syntax"):
        parse_axon_program(source)


def test_parse_return_statement() -> None:
    source = """
lin :: Tensor[B,S,D] -> (Tensor[B,S,D], Tensor[B,S,D])
lin x = do return x, x
"""
    rhs = _parse_rhs_do(source)
    stmt = rhs.body[0]
    assert isinstance(stmt, AxonReturn)
    assert len(stmt.values) == 2


def test_parse_bind_statement() -> None:
    source = """
lin :: Tensor[B,S,D] -> Tensor[B,S,D]
lin x = do
  a, b <- split qkv parts=2
  return a
"""
    rhs = _parse_rhs_do(source)
    stmt = rhs.body[0]
    assert isinstance(stmt, AxonBind)
    assert stmt.targets == ("a", "b")
    assert isinstance(stmt.expr, AxonExprCall)
    assert stmt.expr.callee == "split"


def test_parse_path_literal_single_quoted() -> None:
    source = """
lin :: Tensor[B,S,D] -> Tensor
lin x = do
  y <- Params.param @'model.layers.0.self_attn.q_proj.weight'
  return y
"""
    rhs = _parse_rhs_do(source)
    stmt = rhs.body[0]
    assert isinstance(stmt, AxonBind)
    assert isinstance(stmt.expr, AxonExprCall)
    assert stmt.expr.callee == "Params.param"
    assert len(stmt.expr.args) == 1
    arg0 = stmt.expr.args[0]
    assert isinstance(arg0, AxonExprPath)
    assert arg0.absolute is False
    assert arg0.parts == ("model", "layers", "0", "self_attn", "q_proj", "weight")
