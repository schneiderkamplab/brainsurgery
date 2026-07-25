from __future__ import annotations

import math

from brainsurgery.synapse.axon import (
    elaborate_closed_axon_file,
    flatten_closed_axon_file,
    lower_axon_program_to_graph_ir,
    normalize_closed_axon_file,
    resolve_axon_program_from_path,
    typecheck2_flat_axon_file,
)
from brainsurgery.synapse.axon.codegen2_common import render_python_literal


def test_render_python_literal_handles_nested_nonfinite_floats() -> None:
    source = {
        "limits": (0.0, float("inf"), float("-inf")),
        "nested": [{"value": float("nan")}],
    }

    rendered = render_python_literal(source)
    result = eval(rendered, {"float": float})

    assert result["limits"] == (0.0, float("inf"), float("-inf"))
    assert math.isnan(result["nested"][0]["value"])
    assert "inf" not in rendered.replace('float("inf")', "").replace('float("-inf")', "")


def test_padding_side_pragma_survives_through_graph_ir(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "padding.axon"
    path.write_text(
        """
{-# PADDING_SIDE "right" #-}

main :: Tensor[B,S] -> Tensor[B,S]
main x = x
""",
        encoding="utf-8",
    )

    resolved = resolve_axon_program_from_path(path).ast
    normalized = normalize_closed_axon_file(resolved)
    elaborated = elaborate_closed_axon_file(normalized)
    flat = flatten_closed_axon_file(elaborated)
    typed = typecheck2_flat_axon_file(flat)
    graph = lower_axon_program_to_graph_ir(typed)

    assert graph.pragmas["padding_side"] == "right"
