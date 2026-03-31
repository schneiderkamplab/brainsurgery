from __future__ import annotations

from brainsurgery.synapse.axon.lark_toplevel import (
    ParsedImport,
    ParsedSignature,
    parse_import_line,
    parse_padding_side_pragma,
    parse_signature_line,
)


def test_parse_signature_line() -> None:
    parsed = parse_signature_line("lin@path :: @Path -> Tensor[B,S,Din] -> I -> Tensor[B,S,dim]")
    assert isinstance(parsed, ParsedSignature)
    assert parsed.module_decl == "lin@path"
    assert parsed.type_expr == "@Path -> Tensor[B,S,Din] -> I -> Tensor[B,S,dim]"


def test_parse_import_line_with_parenthesized_members() -> None:
    parsed = parse_import_line("import Activations (gelu_new, silu)")
    assert isinstance(parsed, ParsedImport)
    assert parsed.namespace == "Activations"
    assert parsed.members_tail == "(gelu_new, silu)"


def test_parse_import_line_with_shorthand_members() -> None:
    parsed = parse_import_line("import Activations gelu_new silu")
    assert isinstance(parsed, ParsedImport)
    assert parsed.namespace == "Activations"
    assert parsed.members_tail == "gelu_new silu"


def test_parse_padding_side_pragma_left_and_right() -> None:
    assert parse_padding_side_pragma('{-# PADDING_SIDE "left" #-}') == "left"
    assert parse_padding_side_pragma("{-# PADDING_SIDE 'right' #-}") == "right"


def test_parse_padding_side_pragma_rejects_non_padding_pragma() -> None:
    assert parse_padding_side_pragma("{-# SOMETHING_ELSE 1 #-}") is None
