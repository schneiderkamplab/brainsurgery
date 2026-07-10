from __future__ import annotations

from pathlib import Path

import pytest

from brainsurgery.synapse.axon.ast import (
    AxonBind,
    AxonExprBool,
    AxonExprCall,
    AxonExprNull,
    AxonExprPath,
)
from brainsurgery.synapse.axon.elaborate import elaborate_closed_axon_file
from brainsurgery.synapse.axon.normalize import normalize_closed_axon_file
from brainsurgery.synapse.axon.parse import parse_axon_program
from scripts.axon_elaborate_strong_roundtrip import (
    elaborate_strong_roundtrip_path,
)
from scripts.axon_elaborate_weak_roundtrip import elaborate_weak_roundtrip_path


def _model_axon_paths() -> list[Path]:
    return sorted(Path("brainsurgery/synapse/models").glob("**/*.axon"))


@pytest.mark.parametrize("axon_path", _model_axon_paths(), ids=lambda path: path.as_posix())
def test_elaborate_weak_render_roundtrip_is_canonical(
    axon_path: Path,
    tmp_path: Path,
) -> None:
    assert elaborate_weak_roundtrip_path(axon_path, tmp_path)


@pytest.mark.parametrize("axon_path", _model_axon_paths(), ids=lambda path: path.as_posix())
def test_elaborate_strong_render_roundtrip_is_canonical(
    axon_path: Path,
    tmp_path: Path,
) -> None:
    assert elaborate_strong_roundtrip_path(axon_path, tmp_path)


def test_elaborate_expands_omitted_kwargs_and_optional_args() -> None:
    source = """
use :: Path -> Tensor[B,S,D] -> ?Bool -> ?Int -> Tensor[B,S,D]
use path x ?flag=false ?limit=null = x

main :: Tensor[B,S,D] -> Tensor[B,S,D]
main x = do
  y <- use@proj x
  return y
"""
    normalized = normalize_closed_axon_file(parse_axon_program(source), main_module="main")
    elaborated = elaborate_closed_axon_file(normalized, main_module="main")
    main = next(module for module in elaborated.modules if module.name == "main")
    bind = next(stmt for stmt in main.statements if isinstance(stmt, AxonBind))

    assert isinstance(bind.expr, AxonExprCall)
    assert bind.expr.kwargs == {}
    assert bind.expr.args[-2:] == (AxonExprBool(value=False), AxonExprNull())
    use = next(module for module in elaborated.modules if module.name == "use")
    assert all(param.default_expr is None for param in use.params)


def test_elaborate_scopes_relative_path_defaults_to_callee_path() -> None:
    source = """
use :: Path -> Tensor[B,S,D] -> ?Path -> Tensor[B,S,D]
use path x ?weight_path=@weight = x

main :: Tensor[B,S,D] -> Tensor[B,S,D]
main x = do
  y <- scope@layers do
    z <- use@proj x
    return z
  return y
"""
    normalized = normalize_closed_axon_file(parse_axon_program(source), main_module="main")
    elaborated = elaborate_closed_axon_file(normalized, main_module="main")
    main = next(module for module in elaborated.modules if module.name == "main")
    scope = next(stmt for stmt in main.statements if not isinstance(stmt, AxonBind))
    bind = next(stmt for stmt in scope.body if isinstance(stmt, AxonBind))

    assert isinstance(bind.expr, AxonExprCall)
    assert bind.expr.kwargs == {}
    assert bind.expr.args[-1] == AxonExprPath(absolute=False, parts=("proj", "weight"))
