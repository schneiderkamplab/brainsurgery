from __future__ import annotations

from brainsurgery.synapse.axon.grammar import (
    ParsedBind,
    ParsedFor,
    ParsedReturn,
    ParsedScope,
    ParsedScopeBind,
    parse_statement_head,
)


def test_parse_statement_head_for_range_with_step() -> None:
    parsed = parse_statement_head("for@layers i <- [1..8) step=2 do")
    assert isinstance(parsed, ParsedFor)
    assert parsed.name == "layers"
    assert parsed.var == "i"
    assert parsed.start_delim == "["
    assert parsed.start_expr == "1"
    assert parsed.end_expr == "8"
    assert parsed.end_delim == ")"
    assert parsed.step_expr == "2"


def test_parse_statement_head_scope_bind_without_at() -> None:
    parsed = parse_statement_head("x, y <- scope model.layers do")
    assert isinstance(parsed, ParsedScopeBind)
    assert parsed.raw_targets == "x, y"
    assert parsed.prefix == "model.layers"


def test_parse_statement_head_scope_statement() -> None:
    parsed = parse_statement_head("scope@attn do")
    assert isinstance(parsed, ParsedScope)
    assert parsed.prefix == "attn"


def test_parse_statement_head_return_statement() -> None:
    parsed = parse_statement_head("return x, y")
    assert isinstance(parsed, ParsedReturn)
    assert parsed.raw_values == "x, y"


def test_parse_statement_head_bind_statement() -> None:
    parsed = parse_statement_head("x, y <- split qkv parts=2")
    assert isinstance(parsed, ParsedBind)
    assert parsed.raw_targets == "x, y"
    assert parsed.expr == "split qkv parts=2"
