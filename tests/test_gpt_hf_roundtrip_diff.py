from __future__ import annotations

from pathlib import Path

import pytest

from brainsurgery.engine import create_state_dict_provider
from brainsurgery.engine.execution import _execute_transform_pairs
from brainsurgery.engine.plan import compile_plan


@pytest.fixture
def gpt_model_path(gpt2_local_paths: tuple[Path, Path]) -> Path:
    return gpt2_local_paths[0]


def _run_pipeline(raw_plan: dict[str, object], provider) -> tuple[bool, list[dict[str, object]]]:
    plan = compile_plan(raw_plan)
    return _execute_transform_pairs(
        zip(raw_plan["transforms"], plan.transforms, strict=False),  # type: ignore[index]
        provider,
        interactive=False,
    )


def test_gpt_hf_style_rename_roundtrip_and_diff_no_changes(
    gpt_model_path: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    modern_path = tmp_path / "modern_hf.safetensors"
    modern_rules = [
        (r"legacy::wte\.weight", r"modern::model.embed_tokens.weight"),
        (r"legacy::wpe\.weight", r"modern::model.embed_positions.weight"),
        (r"legacy::ln_f\.(weight|bias)", r"modern::model.norm.\1"),
        (r"legacy::h\.(\d+)\.ln_1\.(weight|bias)", r"modern::model.layers.\1.input_layernorm.\2"),
        (
            r"legacy::h\.(\d+)\.ln_2\.(weight|bias)",
            r"modern::model.layers.\1.post_attention_layernorm.\2",
        ),
        (r"legacy::h\.(\d+)\.attn\.bias", r"modern::model.layers.\1.self_attn.causal_mask"),
        (
            r"legacy::h\.(\d+)\.attn\.c_attn\.(weight|bias)",
            r"modern::model.layers.\1.self_attn.qkv_proj.\2",
        ),
        (
            r"legacy::h\.(\d+)\.attn\.c_proj\.(weight|bias)",
            r"modern::model.layers.\1.self_attn.o_proj.\2",
        ),
        (r"legacy::h\.(\d+)\.mlp\.c_fc\.(weight|bias)", r"modern::model.layers.\1.mlp.up_proj.\2"),
        (
            r"legacy::h\.(\d+)\.mlp\.c_proj\.(weight|bias)",
            r"modern::model.layers.\1.mlp.down_proj.\2",
        ),
    ]
    reverse_rules = [
        (r"modern::model\.embed_tokens\.weight", r"modern::wte.weight"),
        (r"modern::model\.embed_positions\.weight", r"modern::wpe.weight"),
        (r"modern::model\.norm\.(weight|bias)", r"modern::ln_f.\1"),
        (r"modern::model\.layers\.(\d+)\.input_layernorm\.(weight|bias)", r"modern::h.\1.ln_1.\2"),
        (
            r"modern::model\.layers\.(\d+)\.post_attention_layernorm\.(weight|bias)",
            r"modern::h.\1.ln_2.\2",
        ),
        (r"modern::model\.layers\.(\d+)\.self_attn\.causal_mask", r"modern::h.\1.attn.bias"),
        (
            r"modern::model\.layers\.(\d+)\.self_attn\.qkv_proj\.(weight|bias)",
            r"modern::h.\1.attn.c_attn.\2",
        ),
        (
            r"modern::model\.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)",
            r"modern::h.\1.attn.c_proj.\2",
        ),
        (r"modern::model\.layers\.(\d+)\.mlp\.up_proj\.(weight|bias)", r"modern::h.\1.mlp.c_fc.\2"),
        (
            r"modern::model\.layers\.(\d+)\.mlp\.down_proj\.(weight|bias)",
            r"modern::h.\1.mlp.c_proj.\2",
        ),
    ]

    first_pipeline = {
        "inputs": [f"legacy::{gpt_model_path}"],
        "transforms": [{"prefixes": {"mode": "add", "alias": "modern"}}]
        + [{"copy": {"from": source, "to": dest}} for source, dest in modern_rules]
        + [{"save": {"path": str(modern_path), "alias": "modern", "format": "safetensors"}}],
    }

    provider = create_state_dict_provider(
        provider="arena",
        model_paths=compile_plan(first_pipeline).inputs,
        max_io_workers=4,
        arena_root=tmp_path / "arena_first",
        arena_segment_size="2GB",
    )
    try:
        should_continue, executed = _run_pipeline(first_pipeline, provider)
        assert should_continue is True
        assert len(executed) == len(first_pipeline["transforms"])
    finally:
        provider.close()

    assert modern_path.exists()

    second_pipeline = {
        "inputs": [],
        "transforms": [{"load": {"path": str(modern_path), "alias": "modern"}}]
        + [{"move": {"from": source, "to": dest}} for source, dest in reverse_rules]
        + [
            {"load": {"path": str(gpt_model_path), "alias": "original"}},
            {"diff": {"mode": "aliases", "left_alias": "original", "right_alias": "modern"}},
        ],
    }
    provider = create_state_dict_provider(
        provider="arena",
        model_paths={},
        max_io_workers=4,
        arena_root=tmp_path / "arena_second",
        arena_segment_size="2GB",
    )
    try:
        should_continue, executed = _run_pipeline(second_pipeline, provider)
        assert should_continue is True
        assert len(executed) == len(second_pipeline["transforms"])
    finally:
        provider.close()

    output = capsys.readouterr().out
    assert "No differences found." in output
    assert "Missing on left:\n  (none)\n" in output
    assert "Missing on right:\n  (none)\n" in output
    assert "Differing:\n  (none)\n" in output


def test_gpt_reversible_add_and_scale_roundtrip_has_no_diff(
    gpt_model_path: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    edited_path = tmp_path / "edited_ops.safetensors"
    targets = [
        "wpe.weight",
        "ln_f.weight",
        "h.0.attn.c_attn.weight",
        "h.5.mlp.c_fc.bias",
        "h.11.mlp.c_proj.weight",
    ]
    delta = 0.0
    forward_scale = 1.0
    reverse_scale = 1.0

    forward_transforms: list[dict[str, object]] = [
        {"prefixes": {"mode": "add", "alias": "work"}},
        {"copy": {"from": r"original::(.+)", "to": r"work::\1"}},
    ]
    for name in targets:
        delta_name = f"delta.{name}"
        forward_transforms.extend(
            [
                {
                    "fill": {
                        "from": f"work::{name}",
                        "to": f"work::{delta_name}",
                        "mode": "constant",
                        "value": delta,
                    }
                },
                {"add_": {"from": f"work::{delta_name}", "to": f"work::{name}"}},
                {"scale_": {"target": f"work::{name}", "by": forward_scale}},
                {"delete": {"target": f"work::{delta_name}"}},
            ]
        )
    forward_transforms.append(
        {"save": {"path": str(edited_path), "alias": "work", "format": "safetensors"}}
    )

    first_pipeline = {"inputs": [f"original::{gpt_model_path}"], "transforms": forward_transforms}
    provider = create_state_dict_provider(
        provider="arena",
        model_paths=compile_plan(first_pipeline).inputs,
        max_io_workers=4,
        arena_root=tmp_path / "arena_ops_first",
        arena_segment_size="2GB",
    )
    try:
        should_continue, executed = _run_pipeline(first_pipeline, provider)
        assert should_continue is True
        assert len(executed) == len(first_pipeline["transforms"])
    finally:
        provider.close()

    assert edited_path.exists()

    reverse_transforms: list[dict[str, object]] = [
        {"load": {"path": str(edited_path), "alias": "work"}},
    ]
    for name in targets:
        delta_name = f"delta.{name}"
        reverse_transforms.extend(
            [
                {"scale_": {"target": f"work::{name}", "by": reverse_scale}},
                {
                    "fill": {
                        "from": f"work::{name}",
                        "to": f"work::{delta_name}",
                        "mode": "constant",
                        "value": delta,
                    }
                },
                {"subtract_": {"from": f"work::{delta_name}", "to": f"work::{name}"}},
                {"delete": {"target": f"work::{delta_name}"}},
            ]
        )
    reverse_transforms.extend(
        [
            {"load": {"path": str(gpt_model_path), "alias": "original"}},
            {"diff": {"mode": "aliases", "left_alias": "original", "right_alias": "work"}},
        ]
    )

    second_pipeline = {"inputs": [], "transforms": reverse_transforms}
    provider = create_state_dict_provider(
        provider="arena",
        model_paths={},
        max_io_workers=4,
        arena_root=tmp_path / "arena_ops_second",
        arena_segment_size="2GB",
    )
    try:
        should_continue, executed = _run_pipeline(second_pipeline, provider)
        assert should_continue is True
        assert len(executed) == len(second_pipeline["transforms"])
    finally:
        provider.close()

    output = capsys.readouterr().out
    assert "No differences found." in output
    assert "Missing on left:\n  (none)\n" in output
    assert "Missing on right:\n  (none)\n" in output
    assert "Differing:\n  (none)\n" in output
