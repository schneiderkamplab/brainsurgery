from __future__ import annotations

import pytest

from brainsurgery.synapse import (
    TYPING_RULES,
    lower_axon_program_to_synapse_spec,
    parse_axon_program,
    typecheck_axon_program,
)


def test_type_system_rules_are_declared() -> None:
    rule_names = {rule.name for rule in TYPING_RULES}
    assert "T-Var" in rule_names
    assert "T-Call-User" in rule_names
    assert "T-Shape" in rule_names
    assert len(TYPING_RULES) >= 8


def test_typecheck_accepts_well_typed_program() -> None:
    source = """
blk :: Tensor[B,S,D] -> Tensor[B,S,D]
blk x = do
  return x

main :: Tensor[B,S,D] -> Tensor[B,S,D]
main x = do
  y <- blk x
  return y
"""
    modules = parse_axon_program(source)
    signatures = typecheck_axon_program(modules, main_module="main")
    assert "main" in signatures
    assert "blk" in signatures


def test_typecheck_rejects_shape_mismatch_before_lowering() -> None:
    source = """
blk :: Tensor[B,S,768] -> Tensor[B,S,768]
blk x = do
  return x

main :: Tensor[B,S,640] -> Tensor[B,S,768]
main x = do
  y <- blk x
  return y
"""
    modules = parse_axon_program(source)
    with pytest.raises(
        ValueError, match=r"shape mismatch in call 'blk'|shape mismatch in call \"blk\""
    ):
        typecheck_axon_program(modules, main_module="main")


def test_lowering_invokes_typecheck_stage() -> None:
    source = """
blk :: Tensor[B,S,768] -> Tensor[B,S,768]
blk x = do
  return x

main :: Tensor[B,S,640] -> Tensor[B,S,768]
main x = do
  y <- blk x
  return y
"""
    modules = parse_axon_program(source)
    with pytest.raises(ValueError, match=r"Axon typecheck failed"):
        lower_axon_program_to_synapse_spec(modules, main_module="main")


def test_typecheck_allows_sqrt_on_int_input_via_numeric_promotion() -> None:
    source = """
main :: Tensor[B,S,D] -> Tensor[B,S,D]
main x = do
  d <- 640
  s <- sqrt d
  return x
"""
    modules = parse_axon_program(source)
    signatures = typecheck_axon_program(modules, main_module="main")
    assert "main" in signatures
