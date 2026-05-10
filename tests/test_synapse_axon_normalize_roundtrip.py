from __future__ import annotations

from pathlib import Path

import pytest

from brainsurgery.synapse.axon.ast import (
    AxonBind,
    AxonExprAscribe,
    AxonExprBool,
    AxonExprCall,
    AxonExprName,
    AxonExprNull,
    AxonExprPath,
    AxonRepeat,
    AxonYield,
    TypeFloat,
)
from brainsurgery.synapse.axon.normalize import normalize_closed_axon_file
from brainsurgery.synapse.axon.parse import parse_axon_program
from brainsurgery.synapse.axon.validate import validate_normalized_axon_file
from scripts.axon_normalize_weak_roundtrip import normalize_weak_roundtrip_path


def _model_axon_paths() -> list[Path]:
    return sorted(Path("brainsurgery/synapse/models").glob("**/*.axon"))


@pytest.mark.parametrize("axon_path", _model_axon_paths(), ids=lambda path: path.as_posix())
def test_normalize_weak_render_roundtrip_is_canonical(
    axon_path: Path,
    tmp_path: Path,
) -> None:
    assert normalize_weak_roundtrip_path(axon_path, tmp_path)


def test_normalize_expands_path_sugar_without_default_expansion() -> None:
    source = """
use :: Path -> Tensor[B,S,D] -> ?Bool -> ?Int -> Tensor[B,S,D]
use path x ?flag=false ?limit=null = x

main :: Tensor[B,S,D] -> Tensor[B,S,D]
main x = do
  y <- use@proj x
  return y
"""
    normalized = normalize_closed_axon_file(parse_axon_program(source), main_module="main")
    validate_normalized_axon_file(normalized, main_module="main")
    main = next(module for module in normalized.modules if module.name == "main")
    bind = next(stmt for stmt in main.statements if isinstance(stmt, AxonBind))
    call = bind.expr

    assert isinstance(call, AxonExprCall)
    assert call.callee == "use"
    assert call.args[0] == AxonExprPath(absolute=False, parts=("proj",))
    assert "flag" not in call.kwargs
    assert "limit" not in call.kwargs


def test_normalize_desugars_pipe_to_nested_calls() -> None:
    source = """
f :: Tensor[B,S,D] -> Tensor[B,S,D]
f x = x

g :: Tensor[B,S,D] -> ?Bool -> Tensor[B,S,D]
g x ?flag=false = x

main :: Tensor[B,S,D] -> Tensor[B,S,D]
main x = do
  y <- x |> f |> g flag=true
  return y
"""
    normalized = normalize_closed_axon_file(parse_axon_program(source), main_module="main")
    main = next(module for module in normalized.modules if module.name == "main")
    bind = next(stmt for stmt in main.statements if isinstance(stmt, AxonBind))
    outer = bind.expr

    assert isinstance(outer, AxonExprCall)
    assert outer.callee == "g"
    assert outer.kwargs["flag"] == AxonExprBool(value=True)
    inner = outer.args[0]
    assert isinstance(inner, AxonExprCall)
    assert inner.callee == "f"
    assert inner.args == (AxonExprName(name="x"),)


def test_normalize_resolves_zero_arg_definition_references() -> None:
    source = """
EPS = 1e-5

f :: Float -> Float
f x = x

main :: Float -> Float
main x = do
  y <- f EPS
  z <- f (EPS :: Float)
  return z
"""
    normalized = normalize_closed_axon_file(parse_axon_program(source), main_module="main")
    main = next(module for module in normalized.modules if module.name == "main")
    y_bind, z_bind = [stmt for stmt in main.statements if isinstance(stmt, AxonBind)]

    assert isinstance(y_bind.expr, AxonExprCall)
    y_arg = y_bind.expr.args[0]
    assert isinstance(y_arg, AxonExprCall)
    assert y_arg.callee == "EPS"

    assert isinstance(z_bind.expr, AxonExprCall)
    z_arg = z_bind.expr.args[0]
    assert isinstance(z_arg, AxonExprAscribe)
    assert isinstance(z_arg.type_expr, TypeFloat)
    assert isinstance(z_arg.expr, AxonExprCall)
    assert z_arg.expr.callee == "EPS"


def test_normalize_adds_implied_repeat_yield_and_carry() -> None:
    source = """
step :: Tensor[B,S,D] -> Tensor[B,S,D]
step x = x

main :: Tensor[B,S,D] -> Tensor[B,S,D]
main x = do
  x <- for i <- [0..2) do
    x <- step x
  return x
"""
    normalized = normalize_closed_axon_file(parse_axon_program(source), main_module="main")
    main = next(module for module in normalized.modules if module.name == "main")
    repeat = next(stmt for stmt in main.statements if isinstance(stmt, AxonRepeat))

    assert repeat.targets == ("x",)
    assert repeat.carry == ("x",)
    assert isinstance(repeat.body[-1], AxonYield)
    assert repeat.body[-1].values == (AxonExprName(name="x"),)
