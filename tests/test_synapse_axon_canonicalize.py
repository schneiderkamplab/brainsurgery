from __future__ import annotations

from pathlib import Path

from brainsurgery.synapse.axon.ast import render_axon_file
from brainsurgery.synapse.axon.canonicalize import canonicalize_typed_axon_file
from brainsurgery.synapse.axon.flatten import flatten_closed_axon_file
from brainsurgery.synapse.axon.optimize import optimize_flat_typed_axon_file
from brainsurgery.synapse.axon.resolve import resolve_axon_program_from_path
from brainsurgery.synapse.axon.typecheck2 import typecheck2_flat_axon_file


def test_canonicalize_renames_generated_helper_dims_from_callsites() -> None:
    resolved = resolve_axon_program_from_path(
        Path("brainsurgery/synapse/models/gpt2/generic-gpt2-kv.axon")
    ).ast
    flat = flatten_closed_axon_file(resolved, main_module="gpt2")
    typed = typecheck2_flat_axon_file(flat, main_module="gpt2")
    optimized = optimize_flat_typed_axon_file(typed, main_module="gpt2")
    canonical = canonicalize_typed_axon_file(optimized, main_module="gpt2")
    rendered = render_axon_file(canonical, show_types=True)

    assert "gpt2_h2 :: Int -> Tensor[B,S,MODEL_DIM]" in rendered
    assert "gpt2_h1 :: Int -> Tensor[B,S,MODEL_DIM]" in rendered
    assert "Tensor[B,H,K,DH]" in rendered
    assert "Tensor[B,H,P,DH]" in rendered
