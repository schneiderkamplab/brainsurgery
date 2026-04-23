from __future__ import annotations

from brainsurgery.synapse.axon import MaterializeContext, materialize_axon_file, parse_axon_program
from brainsurgery.synapse.axon.ast import render_axon_file


def test_materialize_stage_rewrites_only_config_calls_and_preserves_structure() -> None:
    source = "\n".join(
        [
            "import Config",
            "import Params",
            "",
            'CFG = Config.has_key@text_config ? "text_config" : ""',
            "UNUSED = Config.int@unused default=7",
            "D = Config.int @'{CFG}.hidden_size' default=16",
            'ROOT = Params.root "language_model" default=""',
            "",
            "toy :: Tensor[B,S,D] -> Tensor[B,S,D]",
            "toy x = do",
            "  eps <- Config.float @'{CFG}.rms_norm_eps' default=1e-05",
            "  y <- rmsnorm@norm x eps=eps",
            "  return y",
            "",
        ]
    )
    ast = parse_axon_program(source)
    out = materialize_axon_file(
        ast,
        context=MaterializeContext(
            config={"text_config": {"hidden_size": 32, "rms_norm_eps": 2.5e-5}},
            state_keys=frozenset({"language_model.norm.weight"}),
        ),
    )

    assert out.imports == ("Config", "Params")
    assert "UNUSED" in out.constants

    rendered = render_axon_file(out)
    assert 'CFG = true ? "text_config" : ""' in rendered
    assert "UNUSED = 7" in rendered
    assert "D = 32" in rendered
    assert 'ROOT = Params.root "language_model" default=""' in rendered
    assert "Config.float" not in rendered
    assert "eps <- 2.5e-05" in rendered or "eps <- 2.5e-5" in rendered


def test_materialize_stage_output_is_renderable_and_reparsable() -> None:
    source = "\n".join(
        [
            "import Config",
            "",
            "D = Config.int@hidden_size default=16",
            "",
            "toy :: Tensor[B,S,D] -> Tensor[B,S,D]",
            "toy x = do",
            "  y <- linear@proj x dim=D",
            "  return y",
            "",
        ]
    )
    ast = parse_axon_program(source)
    out = materialize_axon_file(
        ast,
        context=MaterializeContext(
            config={"hidden_size": 24},
            state_keys=frozenset({"proj.weight"}),
        ),
    )
    reparsed = parse_axon_program(render_axon_file(out))
    assert reparsed == out


def test_materialize_stage_supports_config_dim() -> None:
    source = "\n".join(
        [
            "import Config",
            "",
            "D = Config.dim@hidden_size default=16",
            "",
            "toy :: Tensor[B,S,D] -> Tensor[B,S,D]",
            "toy x = x",
            "",
        ]
    )
    ast = parse_axon_program(source)
    out = materialize_axon_file(
        ast,
        context=MaterializeContext(
            config={"hidden_size": 24},
            state_keys=frozenset(),
        ),
    )
    rendered = render_axon_file(out)
    assert "D = 24" in rendered
    assert "Config.dim" not in rendered
