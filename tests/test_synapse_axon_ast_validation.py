from __future__ import annotations

import pytest

from brainsurgery.synapse import (
    AxonModule,
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
    with pytest.raises(ValueError, match="invalid binding target name"):
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
    module = AxonModule(
        name="tiny",
        path_param=None,
        path_params=(),
        params=(),
        returns=(),
        statements=(),
    )
    with pytest.raises(ValueError, match="unknown main module"):
        validate_axon_program((module,), main_module="missing")
