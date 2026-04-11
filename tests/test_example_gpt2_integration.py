from __future__ import annotations

from pathlib import Path

import pytest
import torch

from brainsurgery.engine import (
    RuntimeFlagLifecycleScope,
    create_state_dict_provider,
    reset_runtime_flags_for_scope,
    set_runtime_flag,
)
from brainsurgery.engine.execution import _execute_transform_pairs
from brainsurgery.engine.plan import compile_plan
from brainsurgery.transforms.copy import CopyTransform


@pytest.mark.parametrize("provider_name", ["inmemory", "arena"])
def test_gpt2_copy_tracks_access_counts_for_real_providers(
    tmp_path: Path, provider_name: str, gpt2_local_paths: tuple[Path, Path]
) -> None:
    model_path = gpt2_local_paths[0]
    provider = create_state_dict_provider(
        provider=provider_name,
        model_paths={"loaded": model_path},
        max_io_workers=1,
        arena_root=tmp_path / "arena",
        arena_segment_size="1GB",
    )

    try:
        loaded = provider.get_state_dict("loaded")
        tensor_names = list(loaded.keys())
        assert tensor_names

        provider.get_or_create_alias_state_dict("copied")
        spec = CopyTransform().compile(
            {"from": r"loaded::(.*)", "to": r"copied::\1"},
            default_model=None,
        )
        result = CopyTransform().apply(spec, provider)
        copied = provider.get_state_dict("copied")

        assert result.count == len(tensor_names)
        assert len(copied) == len(tensor_names)

        for name in tensor_names:
            assert loaded.access_counts(name) == {"reads": 1, "writes": 1}
            assert copied.access_counts(name) == {"reads": 0, "writes": 1}
    finally:
        provider.close()


@pytest.mark.parametrize("provider_name", ["inmemory", "arena"])
def test_gpt2_dry_run_pipeline_preserves_loaded_state_dict(
    tmp_path: Path,
    provider_name: str,
    capsys: pytest.CaptureFixture[str],
    gpt2_local_paths: tuple[Path, Path],
) -> None:
    model_path = gpt2_local_paths[0]
    raw = {
        "inputs": [f"loaded::{model_path}"],
        "transforms": [
            {"set": {"dry-run": True, "verbose": True}},
            {"copy": {"from": "ln_f.weight", "to": "ln_f_copy.weight"}},
            {"move": {"from": "^h(.*)", "to": r"i\1"}},
            {"move": {"from": ["i", "*rest"], "to": ["h", "*rest"]}},
            {"assign": {"from": "ln_f_copy.weight", "to": "ln_f.weight"}},
            {"scale": {"from": "ln_f_copy.weight", "to": "ln_f_scaled_equal.weight", "by": 1.0}},
            {"assert": {"equal": {"left": "ln_f.weight", "right": "ln_f_scaled_equal.weight"}}},
            {"scale": {"from": "ln_f_copy.weight", "to": "ln_f_scaled_half.weight", "by": 0.5}},
            {
                "assert": {
                    "not": {"equal": {"left": "ln_f.weight", "right": "ln_f_scaled_half.weight"}}
                }
            },
            {"copy": {"from": "ln_f_copy.weight::[:8]", "to": "demo.x"}},
            {"copy": {"from": "ln_f_copy.weight::[:8]", "to": "demo.y"}},
            {"scale_": {"target": "demo.y", "by": 2.0}},
            {"copy": {"from": "demo.x", "to": "demo.sum"}},
            {"add": {"from_a": "demo.x", "from_b": "demo.x", "to": "demo.sum"}},
            {"fill": {"from": "demo.x", "to": "demo.filled", "mode": "constant", "value": 1.0}},
            {"clamp": {"from": "demo.filled", "to": "demo.clamped", "min": 2.0, "max": 5.0}},
            {"cast": {"from": "demo.clamped", "to": "demo.clamped_cast", "dtype": "float16"}},
            {"reshape": {"from": "demo.clamped", "to": "demo.clamped_2d", "shape": [2, 4]}},
            {"permute": {"from": "demo.clamped_2d", "to": "demo.clamped_perm", "order": [1, 0]}},
            {
                "matmul": {
                    "from_a": "demo.clamped_2d",
                    "from_b": "demo.clamped_perm",
                    "to": "demo.matmul",
                }
            },
            {"copy": {"from": "h.0.attn.c_proj.weight::[:4,:4]", "to": "demo.matrix.weight"}},
            {"phlora_": {"target": "demo.matrix.weight", "rank": 2}},
            {
                "phlora": {
                    "target": "demo.matrix.weight",
                    "target_a": "demo.matrix.a.weight",
                    "target_b": "demo.matrix.b.weight",
                    "rank": 2,
                }
            },
            {"delete": {"target": "ln_f_copy.weight"}},
        ],
    }
    plan = compile_plan(raw)
    provider = create_state_dict_provider(
        provider=provider_name,
        model_paths=plan.inputs,
        max_io_workers=1,
        arena_root=tmp_path / "arena",
        arena_segment_size="1GB",
    )

    probe_names = ["ln_f.weight", "wte.weight", "h.0.attn.c_proj.weight"]
    try:
        loaded = provider.get_state_dict("loaded")
        baseline_tensors = {name: loaded[name].clone() for name in probe_names}
        baseline_counts = {name: loaded.access_counts(name) for name in probe_names}
        baseline_keys = set(loaded.keys())

        reset_runtime_flags_for_scope(RuntimeFlagLifecycleScope.CLI_RUN)
        should_continue, executed = _execute_transform_pairs(
            zip(raw["transforms"], plan.transforms, strict=False),
            provider,
            interactive=False,
        )
        assert should_continue is True
        assert len(executed) == len(raw["transforms"])

        output = capsys.readouterr().out
        assert "dry-run copy: ln_f.weight -> ln_f_copy.weight" in output
        assert "dry-run phlora_" in output

        # Verify persistent provider state is unchanged after leaving dry-run mode.
        set_runtime_flag("dry_run", False)
        loaded = provider.get_state_dict("loaded")
        assert set(loaded.keys()) == baseline_keys
        assert all(not key.startswith("demo.") for key in loaded.keys())
        for name in probe_names:
            assert loaded.access_counts(name) == baseline_counts[name]
            assert torch.equal(loaded[name], baseline_tensors[name]), name
    finally:
        reset_runtime_flags_for_scope(RuntimeFlagLifecycleScope.CLI_RUN)
        provider.close()
