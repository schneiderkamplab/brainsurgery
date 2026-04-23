from __future__ import annotations

from pathlib import Path

import pytest

from brainsurgery.synapse.axon.ast import AxonExprName, AxonFile, AxonModule
from brainsurgery.synapse.axon.parse import parse_axon_program, parse_axon_program_from_path
from brainsurgery.synapse.axon.resolve import resolve_axon_program_from_path
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
            AxonModule(
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
        constants={},
        type_aliases={},
    )
    with pytest.raises(ValueError, match="module imports must be empty"):
        validate_closed_axon_file(program, main_module="main")
