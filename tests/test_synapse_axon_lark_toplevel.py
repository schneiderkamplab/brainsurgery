from __future__ import annotations

from pathlib import Path

import pytest

from brainsurgery.synapse.axon.ast import (
    AxonExprName,
    AxonExprPipe,
    AxonExprTuple,
    TypeAliasDef,
    TypeDim,
    TypePath,
    TypeTensor,
    TypeTuple,
    ast_equal,
    render_axon_file,
    render_type,
)
from brainsurgery.synapse.axon.parse import (
    parse_axon_program,
    parse_axon_program_from_path,
    parse_surface_program_source,
)
from brainsurgery.synapse.axon.resolve import resolve_axon_program_from_path
from brainsurgery.synapse.axon.validate import validate_closed_axon_file


def test_parse_program_source_extracts_signature_type() -> None:
    source = """
lin :: Path -> Tensor[B,S,Din] -> Int -> Tensor[B,S,dim]
lin@path x dim = linear@path x dim=dim bias=true transpose=true
"""
    parsed = parse_surface_program_source(source)
    assert len(parsed.modules) == 1
    signature = parsed.modules[0].signature
    assert signature.definition_decl == "lin"
    sig = signature.type_signature
    assert len(sig.path_params) == 0
    assert tuple(render_type(arg) for arg in sig.arg_types) == ("Path", "Tensor[B,S,Din]", "Int")
    assert render_type(sig.return_type) == "Tensor[B,S,dim]"


def test_parse_program_source_accepts_plain_path_arg_for_path_bound_module() -> None:
    source = """
lin :: Path -> Tensor[B,S,Din] -> Int -> Tensor[B,S,dim]
lin@path x dim = linear@path x dim=dim bias=true transpose=true
"""
    parsed = parse_surface_program_source(source)
    assert len(parsed.modules) == 1
    signature = parsed.modules[0].signature
    sig = signature.type_signature
    assert len(sig.path_params) == 0
    assert tuple(render_type(arg) for arg in sig.arg_types) == (
        "Path",
        "Tensor[B,S,Din]",
        "Int",
    )


def test_parse_program_source_rejects_at_path_in_signature() -> None:
    source = """
lin :: @Path -> Tensor[B,S,Din] -> Int -> Tensor[B,S,dim]
lin@path x dim = linear@path x dim=dim bias=true transpose=true
"""
    with pytest.raises(ValueError, match="invalid Axon source syntax"):
        parse_surface_program_source(source)


def test_parse_program_source_import_parenthesized_members() -> None:
    source = """
import Activations (gelu_new, silu)
lin :: Tensor[B,S,D] -> Tensor[B,S,D]
lin x = x
"""
    parsed = parse_surface_program_source(source)
    assert parsed.imports == ("Activations",)
    assert parsed.imported_members == {"Activations": ("gelu_new", "silu")}


def test_parse_program_source_import_parenthesized_members_allows_trailing_comma() -> None:
    source = """
import Activations (gelu_new, silu,)
lin :: Tensor[B,S,D] -> Tensor[B,S,D]
lin x = x
"""
    parsed = parse_surface_program_source(source)
    assert parsed.imports == ("Activations",)
    assert parsed.imported_members == {"Activations": ("gelu_new", "silu")}


def test_parse_program_source_import_shorthand_members() -> None:
    source = """
import Activations gelu_new silu
lin :: Tensor[B,S,D] -> Tensor[B,S,D]
lin x = x
"""
    parsed = parse_surface_program_source(source)
    assert parsed.imports == ("Activations",)
    assert parsed.imported_members == {"Activations": ("gelu_new", "silu")}


def test_parse_program_source_import_multi_module_specs() -> None:
    source = """
import NN, Attention (reshape_heads, merge_heads), Math exp log
lin :: Tensor[B,S,D] -> Tensor[B,S,D]
lin x = x
"""
    parsed = parse_surface_program_source(source)
    assert parsed.imports == ("NN", "Attention", "Math")
    assert parsed.imported_members == {
        "Attention": ("reshape_heads", "merge_heads"),
        "Math": ("exp", "log"),
    }


def test_parse_program_source_export_symbols() -> None:
    source = """
export (NN, reshape_heads)
lin :: Tensor[B,S,D] -> Tensor[B,S,D]
lin x = x
"""
    parsed = parse_surface_program_source(source)
    assert parsed.exports == ("NN", "reshape_heads")


def test_parse_program_source_padding_side_pragma_left_and_right() -> None:
    left = """
{-# PADDING_SIDE "left" #-}
lin :: Tensor[B,S,D] -> Tensor[B,S,D]
lin x = x
"""
    right = """
{-# PADDING_SIDE 'right' #-}
lin :: Tensor[B,S,D] -> Tensor[B,S,D]
lin x = x
"""
    assert parse_surface_program_source(left).pragmas["padding_side"] == "left"
    assert parse_surface_program_source(right).pragmas["padding_side"] == "right"


def test_parse_program_source_main_pragma_preserved() -> None:
    source = """
{-# MAIN "entry" #-}
entry :: Tensor[B,S,D] -> Tensor[B,S,D]
entry x = x
"""
    assert parse_surface_program_source(source).pragmas["main"] == "entry"


def test_parse_axon_program_adds_implicit_main_pragma_to_ast_rendering() -> None:
    source = """
helper :: Tensor[B,S,D] -> Tensor[B,S,D]
helper x = x

entry :: Tensor[B,S,D] -> Tensor[B,S,D]
entry x = helper x
"""
    program = parse_axon_program(source)
    assert program.pragmas["main"] == "entry"
    assert render_axon_file(program).startswith('{-# MAIN "entry" #-}\n\n')


def test_parse_program_source_checkpoints_pragma_string_normalizes_to_tuple() -> None:
    source = """
{-# CHECKPOINTS "google/gemma-3-270m" #-}
lin :: Tensor[B,S,D] -> Tensor[B,S,D]
lin x = x
"""
    parsed = parse_surface_program_source(source)
    assert parsed.pragmas["checkpoints"] == ("google/gemma-3-270m",)


def test_parse_program_source_checkpoints_pragma_list_preserved() -> None:
    source = """
{-# CHECKPOINTS ["google/gemma-3-270m", "google/gemma-3-270m-it"] #-}
lin :: Tensor[B,S,D] -> Tensor[B,S,D]
lin x = x
"""
    parsed = parse_surface_program_source(source)
    assert parsed.pragmas["checkpoints"] == (
        "google/gemma-3-270m",
        "google/gemma-3-270m-it",
    )


def test_parse_program_source_tokenizer_pragma_preserved() -> None:
    source = """
{-# TOKENIZER "EleutherAI/gpt-neox-20b" #-}
lin :: Tensor[B,S,D] -> Tensor[B,S,D]
lin x = x
"""
    parsed = parse_surface_program_source(source)
    assert parsed.pragmas["tokenizer"] == "EleutherAI/gpt-neox-20b"


def test_parse_program_source_multiple_tokenizer_pragmas_merge() -> None:
    source = """
{-# TOKENIZER "mistralai/Mistral-7B-v0.1" #-}
{-# TOKENIZER ["mistralai/Devstral-Small-2507", "mistralai/Devstral-Small-2507"] #-}
lin :: Tensor[B,S,D] -> Tensor[B,S,D]
lin x = x
"""
    parsed = parse_surface_program_source(source)
    assert parsed.pragmas["tokenizer"] == (
        "mistralai/Mistral-7B-v0.1",
        ("mistralai/Devstral-Small-2507", "mistralai/Devstral-Small-2507"),
    )


def test_parse_program_source_rejects_invalid_import_member_token() -> None:
    source = """
import Activations (gelu-new)
lin :: Tensor[B,S,D] -> Tensor[B,S,D]
lin x = x
"""
    with pytest.raises(ValueError, match="invalid Axon source syntax"):
        parse_surface_program_source(source)


def test_parse_program_source_rejects_invalid_type_expr() -> None:
    source = """
lin :: Tensor -> -> Tensor
lin x = x
"""
    with pytest.raises(ValueError):
        parse_axon_program(source)


def test_parse_program_source_rejects_definition_without_signature() -> None:
    source = "lin@path x = linear@path x dim=4\n"
    with pytest.raises(ValueError, match="invalid Axon source syntax"):
        parse_surface_program_source(source)


def test_parse_program_source_supports_variadic_tensor_dims() -> None:
    source = """
split_like :: Tensor[..S] -> List[Tensor[..S]]
split_like x = split x
"""
    parsed = parse_surface_program_source(source)
    signature = parsed.modules[0].signature
    sig = signature.type_signature
    assert tuple(render_type(arg) for arg in sig.arg_types) == ("Tensor[..S]",)
    assert render_type(sig.return_type) == "List[Tensor[..S]]"


def test_parse_program_source_tuple_type_allows_trailing_comma() -> None:
    source = """
pair :: Tensor[B,S,D] -> (Tensor[B,S,D], Tensor[B,S,D],)
pair x = (x, x)
"""
    parsed = parse_surface_program_source(source)
    signature = parsed.modules[0].signature
    return_type = signature.type_signature.return_type
    assert isinstance(return_type, TypeTuple)
    assert tuple(render_type(item) for item in return_type.items) == (
        "Tensor[B,S,D]",
        "Tensor[B,S,D]",
    )


def test_parse_program_source_tuple_value_allows_trailing_comma() -> None:
    source = """
pair :: Tensor[B,S,D] -> (Tensor[B,S,D], Tensor[B,S,D])
pair x = (x, x,)
"""
    parsed = parse_surface_program_source(source)
    rhs = parsed.modules[0].definition.rhs
    assert isinstance(rhs, AxonExprTuple)
    assert len(rhs.items) == 2


def test_parse_program_source_allows_qualified_pipe_stage() -> None:
    source = """
main :: Tensor[B,S,D] -> Tensor[B,S,D]
main x = x |> Math.exp
"""
    parsed = parse_surface_program_source(source)
    rhs = parsed.modules[0].definition.rhs
    assert isinstance(rhs, AxonExprPipe)
    assert len(rhs.stages) == 1
    stage = rhs.stages[0]
    assert isinstance(stage, AxonExprName)
    assert stage.name == "Math.exp"


def test_parse_render_parse_roundtrip_gpt2_ast_equal(tmp_path: Path) -> None:
    src = Path("brainsurgery/synapse/models/gpt2/gpt2-kv.axon")
    original = parse_axon_program_from_path(src)
    rendered_path = tmp_path / "gpt2-roundtrip.axon"
    rendered_path.write_text(render_axon_file(original), encoding="utf-8")
    reparsed = parse_axon_program_from_path(rendered_path)
    assert ast_equal(original, reparsed)


def test_render_preserves_semantic_parentheses_in_binary_expression(tmp_path: Path) -> None:
    source = """
main :: Int -> Bool
main i = ((i + 1) % PERIOD) == 0
"""
    original = parse_axon_program(source)
    rendered_path = tmp_path / "binary-parens-roundtrip.axon"
    rendered_path.write_text(render_axon_file(original), encoding="utf-8")
    rendered = rendered_path.read_text(encoding="utf-8")
    assert "((i + 1) % PERIOD) == 0" in rendered
    reparsed = parse_axon_program_from_path(rendered_path)
    assert ast_equal(original, reparsed)


def test_render_axon_file_preserves_type_aliases(tmp_path: Path) -> None:
    source = """
type Pair[B,S,D] = (Tensor[B,S,D], Tensor[B,S,D])

main :: Tensor[B,S,D] -> Pair[B,S,D]
main x = (x, x)
"""
    original = parse_axon_program(source)
    rendered_path = tmp_path / "type-alias-roundtrip.axon"
    rendered_path.write_text(render_axon_file(original), encoding="utf-8")
    rendered = rendered_path.read_text(encoding="utf-8")
    assert "type Pair[B, S, D] = (Tensor[B,S,D], Tensor[B,S,D])" in rendered
    reparsed = parse_axon_program_from_path(rendered_path)
    assert ast_equal(original, reparsed)


def test_validate_resolves_qualified_type_alias(tmp_path: Path) -> None:
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
    closed = resolve_axon_program_from_path(root).ast
    validate_closed_axon_file(closed)
    assert {module.name for module in closed.modules} >= {"main"}


def test_validate_resolves_tokenids_alias(tmp_path: Path) -> None:
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
    closed = resolve_axon_program_from_path(root).ast
    validate_closed_axon_file(closed)
    assert {module.name for module in closed.modules} >= {"main"}


def test_parse_program_source_supports_parametric_type_aliases_and_dedicated_path_dim() -> None:
    source = """
type Pair[B,S,D] = (Tensor[B,S,D], Tensor[B,S,D])

main :: Path -> Dim -> Pair[X,Y,Z]
main path d = (path, d)
"""
    parsed = parse_surface_program_source(source)
    alias_def = parsed.type_aliases["Pair"]
    assert alias_def == TypeAliasDef(
        params=("B", "S", "D"),
        value=TypeTuple(
            items=(
                TypeTensor(base="Tensor", dims=("B", "S", "D")),
                TypeTensor(base="Tensor", dims=("B", "S", "D")),
            )
        ),
    )
    signature = parsed.modules[0].signature.type_signature
    assert isinstance(signature.arg_types[0], TypePath)
    assert isinstance(signature.arg_types[1], TypeDim)
    assert render_type(signature.return_type) == "Pair[X,Y,Z]"
