from __future__ import annotations

import pytest

from brainsurgery.synapse.axon.grammar import ParsedSignature, parse_program_source
from brainsurgery.synapse.axon.parser import parse_axon_program
from brainsurgery.synapse.axon.type_system import render_type


def test_parse_program_source_extracts_signature_type() -> None:
    source = """
lin :: @Path -> Tensor[B,S,Din] -> Int -> Tensor[B,S,dim]
lin@path x dim = linear@path x dim=dim bias=true transpose=true
"""
    parsed = parse_program_source(source)
    assert len(parsed.modules) == 1
    signature = parsed.modules[0].signature
    assert isinstance(signature, ParsedSignature)
    assert signature.module_decl == "lin"
    sig = signature.type_signature
    assert len(sig.path_params) == 1
    assert sig.path_params[0].name is None
    assert render_type(sig.path_params[0].type_expr) == "Path"
    assert tuple(render_type(arg) for arg in sig.arg_types) == ("Tensor[B,S,Din]", "Int")
    assert render_type(sig.return_type) == "Tensor[B,S,dim]"


def test_parse_program_source_import_parenthesized_members() -> None:
    source = """
import Activations (gelu_new, silu)
lin :: Tensor[B,S,D] -> Tensor[B,S,D]
lin x = x
"""
    parsed = parse_program_source(source)
    assert parsed.imports == ("Activations",)
    assert parsed.imported_members == {"Activations": ("gelu_new", "silu")}


def test_parse_program_source_import_shorthand_members() -> None:
    source = """
import Activations gelu_new silu
lin :: Tensor[B,S,D] -> Tensor[B,S,D]
lin x = x
"""
    parsed = parse_program_source(source)
    assert parsed.imports == ("Activations",)
    assert parsed.imported_members == {"Activations": ("gelu_new", "silu")}


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
    assert parse_program_source(left).pragmas["padding_side"] == "left"
    assert parse_program_source(right).pragmas["padding_side"] == "right"


def test_parse_program_source_checkpoints_pragma_string_normalizes_to_tuple() -> None:
    source = """
{-# CHECKPOINTS "google/gemma-3-270m" #-}
lin :: Tensor[B,S,D] -> Tensor[B,S,D]
lin x = x
"""
    parsed = parse_program_source(source)
    assert parsed.pragmas["checkpoints"] == ("google/gemma-3-270m",)


def test_parse_program_source_checkpoints_pragma_list_preserved() -> None:
    source = """
{-# CHECKPOINTS ["google/gemma-3-270m", "google/gemma-3-270m-it"] #-}
lin :: Tensor[B,S,D] -> Tensor[B,S,D]
lin x = x
"""
    parsed = parse_program_source(source)
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
    parsed = parse_program_source(source)
    assert parsed.pragmas["tokenizer"] == "EleutherAI/gpt-neox-20b"


def test_parse_program_source_multiple_tokenizer_pragmas_merge() -> None:
    source = """
{-# TOKENIZER "mistralai/Mistral-7B-v0.1" #-}
{-# TOKENIZER ["mistralai/Devstral-Small-2507", "mistralai/Devstral-Small-2507"] #-}
lin :: Tensor[B,S,D] -> Tensor[B,S,D]
lin x = x
"""
    parsed = parse_program_source(source)
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
        parse_program_source(source)


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
        parse_program_source(source)


def test_parse_program_source_supports_variadic_tensor_dims() -> None:
    source = """
split_like :: Tensor[..S] -> List[Tensor[..S]]
split_like x = split x
"""
    parsed = parse_program_source(source)
    signature = parsed.modules[0].signature
    sig = signature.type_signature
    assert tuple(render_type(arg) for arg in sig.arg_types) == ("Tensor[..S]",)
    assert render_type(sig.return_type) == "List[Tensor[..S]]"
