from __future__ import annotations

import pytest

from brainsurgery.synapse import (
    AxonDefinition,
    parse_axon_program,
    validate_axon_program,
)


def test_ast_validation_rejects_duplicate_parameter_names() -> None:
    source = """
tiny :: Tensor -> Tensor -> Tensor
tiny x x = do
  return x
"""
    with pytest.raises(ValueError, match="duplicate parameter name"):
        parse_axon_program(source)


def test_ast_validation_rejects_invalid_binding_target_name() -> None:
    source = """
tiny :: Tensor -> Tensor
tiny x = do
  1y <- x
  return x
"""
    with pytest.raises(ValueError, match="invalid Axon source syntax"):
        parse_axon_program(source)


def test_ast_validation_rejects_signature_return_arity_mismatch() -> None:
    source = """
tiny :: Tensor -> (Tensor, Tensor)
tiny x = do
  return x, x, x
"""
    with pytest.raises(ValueError, match="return arity mismatch"):
        parse_axon_program(source)


def test_ast_validation_rejects_scope_bind_without_compatible_return() -> None:
    source = """
tiny :: Tensor -> Tensor
tiny x = do
  y <- scope@attn do
    z <- x
  return y
"""
    with pytest.raises(ValueError, match="scope bind requires a reachable return"):
        parse_axon_program(source)


def test_ast_validation_rejects_unknown_main_module() -> None:
    module = AxonDefinition(
        name="tiny",
        path_param=None,
        path_params=(),
        params=(),
        returns=(),
        statements=(),
    )
    with pytest.raises(ValueError, match="unknown main module"):
        validate_axon_program((module,), main_module="missing")


def test_ast_validation_rejects_yield_outside_for() -> None:
    source = """
tiny :: Tensor -> Tensor
tiny x = do
  yield x
  return x
"""
    with pytest.raises(ValueError, match="yield is only valid inside for-loop bodies"):
        parse_axon_program(source)


def test_ast_validation_rejects_non_terminal_yield_in_for_body() -> None:
    source = """
tiny :: Tensor -> Tensor
tiny x = do
  y <- for i <- [0..2) carry (x) do
    yield x
    x <- x
  return y
"""
    with pytest.raises(ValueError, match="yield must be the final statement in a for-loop body"):
        parse_axon_program(source)


def test_ast_validation_allows_scope_return_inside_for_body() -> None:
    source = """
tiny :: Tensor[B,S] -> Tensor[B,S]
tiny x = do
  y <- for i <- [0..2) carry (x) do
    z <- scope@layer do
      return x
    x <- z
    yield x
  return y
"""
    parse_axon_program(source)
