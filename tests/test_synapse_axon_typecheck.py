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
main :: Tensor[B,S,D] -> Tensor[B,S,D]
main x = do
  d <- 640
  s <- Math.log d
  return x
"""
    modules = _parse_from_tmp_source(tmp_path, source)
    signatures = typecheck_axon_program(modules, main_module="main")
    assert "main" in signatures


def test_typecheck_allows_floor_on_int_input_via_numeric_promotion(tmp_path: Path) -> None:
    source = """
main :: Tensor[B,S,D] -> Tensor[B,S,D]
main x = do
  d <- 640
  s <- Math.floor d
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
import Tensor

main :: Tensor[B,S] -> Tensor[B,1,S,1]
main x = do
  y <- Tensor.reshape x shape=[B, 1, S, 1]
  return y
"""
    modules = _parse_from_tmp_source(tmp_path, source)
    signatures = typecheck_axon_program(modules, main_module="main")
    assert "main" in signatures


def test_typecheck_allows_unsqueeze_via_prelude_exported_tensor_namespace(
    tmp_path: Path,
) -> None:
    source = """
import Tensor

main :: Tensor[B,S] -> Tensor[B,1,S,1]
main x = do
  y <- Tensor.unsqueeze x dim=1
  z <- Tensor.unsqueeze y dim=3
  return z
"""
    modules = _parse_from_tmp_source(tmp_path, source)
    signatures = typecheck_axon_program(modules, main_module="main")
    assert "main" in signatures


def test_typecheck_allows_variadic_split_on_4d_tensor(tmp_path: Path) -> None:
    source = """
import Tensor

split_any :: Tensor[..S] -> List[Tensor[..S]]
split_any x = Tensor.split x dim=-1 sizes=[64, 64]

main :: Tensor[B,H,T,HD] -> Tensor[B,H,T,HD]
main x = do
  a, b <- split_any x
  y <- Tensor.concat a b dim=-1
  return y
"""
    modules = _parse_from_tmp_source(tmp_path, source)
    signatures = typecheck_axon_program(modules, main_module="main")
    assert "main" in signatures


def test_typecheck_allows_extra_path_suffix_binding_for_path_params(tmp_path: Path) -> None:
    source = """
my_linear :: Path -> Tensor[B,S,D] -> ?Path -> ?Path -> Tensor[B,S,D]
my_linear@path x ?weight_path=@weight ?bias_path=@bias = do
  return x

main :: Tensor[B,S,D] -> Tensor[B,S,D]
main x = do
  y <- my_linear@q_proj@weight@bias x
  return y
"""
    modules = _parse_from_tmp_source(tmp_path, source)
    signatures = typecheck_axon_program(modules, main_module="main")
    assert "main" in signatures


def test_typecheck_rejects_too_many_extra_path_suffixes(tmp_path: Path) -> None:
    source = """
my_linear :: Path -> Tensor[B,S,D] -> ?Path -> Tensor[B,S,D]
my_linear@path x ?weight_path=@weight = do
  return x

main :: Tensor[B,S,D] -> Tensor[B,S,D]
main x = do
  y <- my_linear@q_proj@weight@bias x
  return y
"""
    modules = _parse_from_tmp_source(tmp_path, source)
    with pytest.raises(ValueError, match="extra @path suffixes"):
        typecheck_axon_program(modules, main_module="main")


def test_typecheck_allows_plain_path_signature_for_path_bound_module(tmp_path: Path) -> None:
    source = """
my_linear :: Path -> Tensor[B,S,D] -> ?Path -> ?Path -> Tensor[B,S,D]
my_linear@path x ?weight_path=@weight ?bias_path=@bias = do
  return x

main :: Tensor[B,S,D] -> Tensor[B,S,D]
main x = do
  y <- my_linear@q_proj@weight@bias x
  return y
"""
    modules = _parse_from_tmp_source(tmp_path, source)
    signatures = typecheck_axon_program(modules, main_module="main")
    assert "main" in signatures


def test_typecheck_and_lowering_allow_value_level_dim_symbol_from_alias(tmp_path: Path) -> None:
    source = """
type CacheLayer = (Tensor[B,H,T,DH], Tensor[B,H,T,DH])
type Cache = List[CacheLayer]

past_length :: ?Cache -> Int
past_length cache = (cache == null) ? 0 : T

main :: ?Cache -> Int
main cache = past_length cache
"""
    modules = _parse_from_tmp_source(tmp_path, source)
    signatures = typecheck_axon_program(modules, main_module="main")
    assert "past_length" in signatures
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    block_graph = spec["model"]["blocks"]["past_length"]["graph"]

    def _contains_t_expr(value: object) -> bool:
        if isinstance(value, dict):
            if value.get("_op") == "_ir_expr" and value.get("value") == "T":
                return True
            return any(_contains_t_expr(item) for item in value.values())
        if isinstance(value, list):
            return any(_contains_t_expr(item) for item in value)
        return False

    assert _contains_t_expr(block_graph)


def test_lowering_infers_split_sizes_from_bind_arity(tmp_path: Path) -> None:
    source = """
import Tensor

main :: Tensor[B,S,12] -> Tensor[B,S,12]
main x = do
  a, b, c <- Tensor.split x
  return a
"""
    modules = _parse_from_tmp_source(tmp_path, source)
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    split_calls = [
        node
        for item in spec["model"]["graph"]
        for node in item.values()
        if isinstance(node, dict)
        and node.get("_op") == "call"
        and node.get("_target") == "Tensor.split"
    ]
    assert len(split_calls) == 1
    assert split_calls[0].get("_args") == "x"
    assert split_calls[0].get("dim") == -1
    assert split_calls[0].get("sizes") == [4, 4, 4]


def test_lowering_infers_chunk_parts_from_bind_arity(tmp_path: Path) -> None:
    source = """
import Tensor

main :: Tensor[B,S,12] -> Tensor[B,S,12]
main x = do
  a, b, c <- Tensor.chunk x
  return a
"""
    modules = _parse_from_tmp_source(tmp_path, source)
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    chunk_calls = [
        node
        for item in spec["model"]["graph"]
        for node in item.values()
        if isinstance(node, dict)
        and node.get("_op") == "call"
        and node.get("_target") == "Tensor.chunk"
    ]
    assert len(chunk_calls) == 1
    assert chunk_calls[0].get("_args") == "x"
    assert chunk_calls[0].get("dim") == -1
    assert chunk_calls[0].get("parts") == 3


def test_lowering_keeps_explicit_split_sizes_even_if_bind_arity_differs(tmp_path: Path) -> None:
    source = """
import Tensor

main :: Tensor[B,S,12] -> Tensor[B,S,12]
main x = do
  a, b, c <- Tensor.split x sizes=[6, 6]
  return a
"""
    modules = _parse_from_tmp_source(tmp_path, source)
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    split_calls = [
        node
        for item in spec["model"]["graph"]
        for node in item.values()
        if isinstance(node, dict)
        and node.get("_op") == "call"
        and node.get("_target") == "Tensor.split"
    ]
    assert len(split_calls) == 1
    assert split_calls[0].get("sizes") == [6, 6]


def test_lowering_emits_for_carry_yield_and_bind_metadata(tmp_path: Path) -> None:
    source = """
main :: Tensor[B,S,D] -> Tensor[B,S,D]
main x = do
  y <- for@layers i <- [0..2) carry (x) do
    x <- x
    yield x
  return y
"""
    modules = _parse_from_tmp_source(tmp_path, source)
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    loops = [
        node
        for item in spec["model"]["graph"]
        for node in item.values()
        if isinstance(node, dict) and node.get("_op") == "for"
    ]
    assert len(loops) == 1
    loop = loops[0]
    assert loop.get("_carry") == ["x"]
    assert loop.get("_yield") == [{"_expr": "name", "id": "x"}]
    assert loop.get("_bind") == ["y"]


def test_import_uses_exported_namespace_symbols_from_dependency(tmp_path: Path) -> None:
    helper_path = tmp_path / "Helper.axon"
    helper_path.write_text(
        """
export helper

helper :: Tensor[B,S,D] -> Tensor[B,S,D]
helper x = x
""",
        encoding="utf-8",
    )
    layer_path = tmp_path / "Layer.axon"
    layer_path.write_text(
        """
import Helper
export Helper

layer :: Tensor[B,S,D] -> Tensor[B,S,D]
layer x = Helper.helper x
""",
        encoding="utf-8",
    )
    root_path = tmp_path / "test.axon"
    root_path.write_text(
        """
import Layer

main :: Tensor[B,S,D] -> Tensor[B,S,D]
main x = do
  y <- Helper.helper x
  return y
""",
        encoding="utf-8",
    )
    modules = parse_axon_program_from_path(root_path)
    signatures = typecheck_axon_program(modules, main_module="main")
    assert "main" in signatures


def test_prelude_exports_core_namespaces_without_explicit_imports(tmp_path: Path) -> None:
    source = """
main :: Path -> Tensor[B,S,D] -> Tensor[B,1,S,D]
main@path x = do
  y <- NN.rmsnorm@path x
  z <- Tensor.unsqueeze y 1
  w <- Math.exp z
  return w
"""
    modules = _parse_from_tmp_source(tmp_path, source)
    signatures = typecheck_axon_program(modules, main_module="main")
    assert "main" in signatures


def test_import_namespace_allows_only_exported_members(tmp_path: Path) -> None:
    source = """
import Activations

main :: Tensor[B,S,D] -> Tensor[B,S,D]
main x = do
  y <- Activations.gegelu_limit x
  return y
"""
    modules = _parse_from_tmp_source(tmp_path, source)
    with pytest.raises(
        ValueError,
        match=r"unknown callee 'Activations\.gegelu_limit'|unknown callee \"Activations\.gegelu_limit\"",
    ):
        typecheck_axon_program(modules, main_module="main")


def test_import_namespace_allows_exported_members(tmp_path: Path) -> None:
    source = """
import Config

main :: Tensor[B,S,D] -> Tensor[B,S,D]
main x = do
  d <- Config.int@hidden_size default=768
  return x
"""
    modules = _parse_from_tmp_source(tmp_path, source)
    signatures = typecheck_axon_program(modules, main_module="main")
    assert "main" in signatures
