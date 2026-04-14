from __future__ import annotations

from pathlib import Path

import pytest

from brainsurgery.synapse import (
    TYPING_RULES,
    lower_axon_program_to_synapse_spec,
    parse_axon_program,
    parse_axon_program_from_path,
    typecheck_axon_program,
)


def _parse_from_tmp_source(tmp_path: Path, source: str) -> tuple:
    axon_path = tmp_path / "test.axon"
    axon_path.write_text(source, encoding="utf-8")
    return parse_axon_program_from_path(axon_path)


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


def test_typecheck_allows_log_on_int_input_via_numeric_promotion(tmp_path: Path) -> None:
    source = """
import Math

main :: Tensor[B,S,D] -> Tensor[B,S,D]
main x = do
  d <- 640
  s <- log d
  return x
"""
    modules = _parse_from_tmp_source(tmp_path, source)
    signatures = typecheck_axon_program(modules, main_module="main")
    assert "main" in signatures


def test_typecheck_allows_floor_on_int_input_via_numeric_promotion(tmp_path: Path) -> None:
    source = """
import Math

main :: Tensor[B,S,D] -> Tensor[B,S,D]
main x = do
  d <- 640
  s <- floor d
  return x
"""
    modules = _parse_from_tmp_source(tmp_path, source)
    signatures = typecheck_axon_program(modules, main_module="main")
    assert "main" in signatures


def test_typecheck_allows_broadcasted_tensor_binary_mul() -> None:
    source = """
main :: Tensor[B,1,S,1] -> Tensor[B,H,S,D] -> Tensor[B,H,S,D]
main scale q = do
  y <- scale * q
  return y
"""
    modules = parse_axon_program(source)
    signatures = typecheck_axon_program(modules, main_module="main")
    assert "main" in signatures


def test_typecheck_allows_generic_reshape_to_higher_rank(tmp_path: Path) -> None:
    source = """
import Prelude

main :: Tensor[B,S] -> Tensor[B,1,S,1]
main x = do
  y <- reshape x shape=[B, 1, S, 1]
  return y
"""
    modules = _parse_from_tmp_source(tmp_path, source)
    signatures = typecheck_axon_program(modules, main_module="main")
    assert "main" in signatures


def test_typecheck_rejects_unsqueeze_outside_builtins(tmp_path: Path) -> None:
    source = """
import Prelude

main :: Tensor[B,S] -> Tensor[B,1,S,1]
main x = do
  y <- unsqueeze x dim=1
  z <- unsqueeze y dim=3
  return z
"""
    modules = _parse_from_tmp_source(tmp_path, source)
    with pytest.raises(ValueError, match=r"direct primitive call must use _xyz syntax, got 'unsqueeze'"):
        typecheck_axon_program(modules, main_module="main")


def test_typecheck_allows_variadic_split_on_4d_tensor(tmp_path: Path) -> None:
    source = """
import Prelude

split_any :: Tensor[..S] -> List[Tensor[..S]]
split_any x = split x dim=-1 sizes=[64, 64]

main :: Tensor[B,H,T,HD] -> Tensor[B,H,T,HD]
main x = do
  a, b <- split_any x
  y <- concat a b dim=-1
  return y
"""
    modules = _parse_from_tmp_source(tmp_path, source)
    signatures = typecheck_axon_program(modules, main_module="main")
    assert "main" in signatures


def test_lowering_infers_split_sizes_from_bind_arity(tmp_path: Path) -> None:
    source = """
import Prelude

main :: Tensor[B,S,12] -> Tensor[B,S,12]
main x = do
  a, b, c <- split x
  return a
"""
    modules = _parse_from_tmp_source(tmp_path, source)
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    split_calls = [
        node
        for item in spec["model"]["graph"]
        for node in item.values()
        if isinstance(node, dict) and node.get("_op") == "call" and node.get("_target") == "Prelude.split"
    ]
    assert len(split_calls) == 1
    assert split_calls[0].get("_args") == "x"
    assert split_calls[0].get("dim") == -1
    assert split_calls[0].get("sizes") == [4, 4, 4]


def test_lowering_infers_chunk_parts_from_bind_arity(tmp_path: Path) -> None:
    source = """
import Prelude

main :: Tensor[B,S,12] -> Tensor[B,S,12]
main x = do
  a, b, c <- chunk x
  return a
"""
    modules = _parse_from_tmp_source(tmp_path, source)
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    chunk_calls = [
        node
        for item in spec["model"]["graph"]
        for node in item.values()
        if isinstance(node, dict) and node.get("_op") == "call" and node.get("_target") == "Prelude.chunk"
    ]
    assert len(chunk_calls) == 1
    assert chunk_calls[0].get("_args") == "x"
    assert chunk_calls[0].get("dim") == -1
    assert chunk_calls[0].get("parts") == 3


def test_lowering_keeps_explicit_split_sizes_even_if_bind_arity_differs(tmp_path: Path) -> None:
    source = """
import Prelude

main :: Tensor[B,S,12] -> Tensor[B,S,12]
main x = do
  a, b, c <- split x sizes=[6, 6]
  return a
"""
    modules = _parse_from_tmp_source(tmp_path, source)
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    split_calls = [
        node
        for item in spec["model"]["graph"]
        for node in item.values()
        if isinstance(node, dict) and node.get("_op") == "call" and node.get("_target") == "Prelude.split"
    ]
    assert len(split_calls) == 1
    assert split_calls[0].get("sizes") == [6, 6]
