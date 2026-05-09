from __future__ import annotations

from pathlib import Path

import pytest

from brainsurgery.synapse.axon.ast import AxonExprName, AxonFile, AxonDefinition
from brainsurgery.synapse.axon.entrypoint import resolve_main_module
from brainsurgery.synapse.axon.normalize import normalize_closed_axon_file
from brainsurgery.synapse.axon.parse import parse_axon_program, parse_axon_program_from_path
from brainsurgery.synapse.axon.resolve import prune_unreachable_definitions, resolve_axon_program_from_path
from brainsurgery.synapse.axon.validate import validate_closed_axon_file


def _resolved_from_tmp_source(tmp_path: Path, source: str) -> AxonFile:
    axon_path = tmp_path / "test.axon"
    axon_path.write_text(source, encoding="utf-8")
    return resolve_axon_program_from_path(axon_path).ast


def test_validate_closed_accepts_resolved_program() -> None:
    source = """
blk :: Tensor[B,S,D] -> Tensor[B,S,D]
blk x = do
  return x

main :: Tensor[B,S,D] -> Tensor[B,S,D]
main x = do
  y <- blk x
  return y
"""
    program = parse_axon_program(source)
    validate_closed_axon_file(program, main_module="main")


def test_main_pragma_selects_entry_definition_when_no_explicit_main() -> None:
    source = """
{-# MAIN "entry" #-}
helper :: Tensor[B,S,D] -> Tensor[B,S,D]
helper x = x

entry :: Tensor[B,S,D] -> Tensor[B,S,D]
entry x = helper x

not_entry :: Tensor[B,S,D] -> Tensor[B,S,D]
not_entry x = x
"""
    program = normalize_closed_axon_file(parse_axon_program(source))
    validate_closed_axon_file(program)
    assert resolve_main_module(program) == "entry"


def test_explicit_main_overrides_main_pragma() -> None:
    source = """
{-# MAIN "entry" #-}
entry :: Tensor[B,S,D] -> Tensor[B,S,D]
entry x = x

other :: Tensor[B,S,D] -> Tensor[B,S,D]
other x = x
"""
    program = normalize_closed_axon_file(parse_axon_program(source))
    validate_closed_axon_file(program, main_module="other")
    assert resolve_main_module(program, main_module="other") == "other"


def test_validate_closed_rejects_unknown_main_pragma() -> None:
    source = """
{-# MAIN "missing" #-}
entry :: Tensor[B,S,D] -> Tensor[B,S,D]
entry x = x
"""
    program = parse_axon_program(source)
    with pytest.raises(ValueError, match="MAIN pragma references unknown definition"):
        validate_closed_axon_file(program)


def test_normalize_rejects_invalid_main_pragma() -> None:
    source = """
{-# MAIN ["entry"] #-}
entry :: Tensor[B,S,D] -> Tensor[B,S,D]
entry x = x
"""
    with pytest.raises(ValueError, match="MAIN.*non-empty string"):
        normalize_closed_axon_file(parse_axon_program(source))


def test_validate_closed_rejects_non_closed_axon_file(tmp_path: Path) -> None:
    source = """
import Math

main :: Tensor[B,S,D] -> Tensor[B,S,D]
main x = Math.exp x
"""
    axon_path = tmp_path / "test.axon"
    axon_path.write_text(source, encoding="utf-8")
    program = parse_axon_program_from_path(axon_path)
    with pytest.raises(ValueError, match="closed AST must not carry file imports"):
        validate_closed_axon_file(program, main_module="main")


def test_prune_keeps_value_referenced_only_by_path_template(tmp_path: Path) -> None:
    source = """
{-# MAIN "main" #-}
import Config

CFG = Config.has_key @@text_config ? "text_config." : ""
MODEL_DIM = Config.int @@'{CFG}hidden_size' default=128

main :: Tensor[B,S,D] -> Tensor[B,S,D]
main x = do
  _ <- MODEL_DIM
  return x
"""
    program = _resolved_from_tmp_source(tmp_path, source)
    pruned = prune_unreachable_definitions(program)
    names = {module.name for module in pruned.modules}
    assert "CFG" in names
    assert "MODEL_DIM" in names


def test_validate_closed_rejects_unresolved_name_in_body() -> None:
    source = """
main :: Tensor[B,S,D] -> Tensor[B,S,D]
main x = do
  return missing
"""
    program = parse_axon_program(source)
    with pytest.raises(ValueError, match="unresolved name 'missing'"):
        validate_closed_axon_file(program, main_module="main")


def test_validate_closed_accepts_value_level_dim_symbol_from_alias(tmp_path: Path) -> None:
    source = """
type CacheLayer[B,H,T,DH] = (Tensor[B,H,T,DH], Tensor[B,H,T,DH])
type Cache[B,H,T,DH] = List[CacheLayer[B,H,T,DH]]

past_length :: ?Cache[B,H,T,DH] -> Int
past_length cache = (cache == null) ? 0 : T

main :: ?Cache[B,H,T,DH] -> Int
main cache = past_length cache
"""
    program = _resolved_from_tmp_source(tmp_path, source)
    validate_closed_axon_file(program, main_module="main")


def test_validate_closed_resolves_qualified_type_alias(tmp_path: Path) -> None:
    lib_source = """
type CacheLayer[B,H,T,DH] = (Tensor[B,H,T,DH], Tensor[B,H,T,DH])
type Cache[B,H,T,DH] = List[CacheLayer[B,H,T,DH]]

Cache.marker :: Int -> Int
Cache.marker x = x
"""
    main_source = """
import Cache

main :: Cache.Cache[B,H,T,DH] -> Cache.Cache[B,H,T,DH]
main x = x
"""
    (tmp_path / "Cache.axon").write_text(lib_source, encoding="utf-8")
    root = tmp_path / "main.axon"
    root.write_text(main_source, encoding="utf-8")
    validate_closed_axon_file(resolve_axon_program_from_path(root).ast)


def test_validate_closed_resolves_tokenids_alias(tmp_path: Path) -> None:
    lib_source = """
type TokenIds[B,S] = Tensor[B,S]

Tensor.marker :: Int -> Int
Tensor.marker x = x
"""
    main_source = """
import Tensor

main :: TokenIds[X,Y] -> Tensor[X,Y]
main x = x
"""
    (tmp_path / "Tensor.axon").write_text(lib_source, encoding="utf-8")
    root = tmp_path / "main.axon"
    root.write_text(main_source, encoding="utf-8")
    validate_closed_axon_file(resolve_axon_program_from_path(root).ast)


def test_validate_closed_rejects_bad_type_alias_arity() -> None:
    source = """
type Pair[B,S] = Tensor[B,S]

main :: Pair[X] -> Pair[X]
main x = x
"""
    program = parse_axon_program(source)
    with pytest.raises(ValueError, match="expects 2 args, got 1"):
        validate_closed_axon_file(program, main_module="main")


def test_validate_closed_rejects_scope_placeholder_without_binding() -> None:
    source = """
main :: Tensor[B,S,D] -> Tensor[B,S,D]
main x = do
  y <- scope@'layers.{i}' do
    return x
  return y
"""
    program = parse_axon_program(source)
    with pytest.raises(ValueError, match="unresolved scope placeholder 'i'"):
        validate_closed_axon_file(program, main_module="main")


def test_validate_closed_rejects_module_level_import_state() -> None:
    program = AxonFile(
        modules=(
            AxonDefinition(
                name="main",
                path_param=None,
                path_params=(),
                params=(),
                returns=(),
                statements=(),
                body_expr=AxonExprName(name="main"),
                imports=("Math",),
            ),
        ),
        imports=(),
        imported_members={},
        exports=(),
        pragmas={},
        type_aliases={},
    )
    with pytest.raises(ValueError, match="module imports must be empty"):
        validate_closed_axon_file(program, main_module="main")
