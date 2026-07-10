from __future__ import annotations

import brainsurgery


def test_main_passes_explicit_subcommand_through(monkeypatch) -> None:
    calls: list[tuple[list[str], str]] = []

    def _fake_app(*, args, prog_name):  # type: ignore[no-untyped-def]
        calls.append((list(args), prog_name))

    monkeypatch.setattr(brainsurgery, "app", _fake_app)

    brainsurgery.main(["cli", "examples/gpt2.yaml"])

    assert calls == [(["cli", "examples/gpt2.yaml"], "brainsurgery")]


def test_main_passes_empty_args_without_defaulting(monkeypatch) -> None:
    calls: list[tuple[list[str], str]] = []

    def _fake_app(*, args, prog_name):  # type: ignore[no-untyped-def]
        calls.append((list(args), prog_name))

    monkeypatch.setattr(brainsurgery, "app", _fake_app)

    brainsurgery.main(["webcli", "--port", "9000"])

    assert calls == [(["webcli", "--port", "9000"], "brainsurgery")]


def test_main_defaults_to_cli_when_no_subcommand(monkeypatch) -> None:
    calls: list[tuple[list[str], str]] = []

    def _fake_app(*, args, prog_name):  # type: ignore[no-untyped-def]
        calls.append((list(args), prog_name))

    monkeypatch.setattr(brainsurgery, "app", _fake_app)

    brainsurgery.main(["examples/gpt2.yaml"])

    assert calls == [(["cli", "examples/gpt2.yaml"], "brainsurgery")]


def test_main_preserves_explicit_webui_subcommand(monkeypatch) -> None:
    calls: list[tuple[list[str], str]] = []

    def _fake_app(*, args, prog_name):  # type: ignore[no-untyped-def]
        calls.append((list(args), prog_name))

    monkeypatch.setattr(brainsurgery, "app", _fake_app)

    brainsurgery.main(["webui", "--port", "9010"])

    assert calls == [(["webui", "--port", "9010"], "brainsurgery")]


def test_main_preserves_explicit_synapse_subcommand(monkeypatch) -> None:
    calls: list[tuple[list[str], str]] = []

    def _fake_app(*, args, prog_name):  # type: ignore[no-untyped-def]
        calls.append((list(args), prog_name))

    monkeypatch.setattr(brainsurgery, "app", _fake_app)

    brainsurgery.main(["synapse", "emit", "spec.yaml", "out.py"])

    assert calls == [(["synapse", "emit", "spec.yaml", "out.py"], "brainsurgery")]


def test_main_reorders_cli_options_after_config_items(monkeypatch) -> None:
    calls: list[tuple[list[str], str]] = []

    def _fake_app(*, args, prog_name):  # type: ignore[no-untyped-def]
        calls.append((list(args), prog_name))

    monkeypatch.setattr(brainsurgery, "app", _fake_app)

    brainsurgery.main(["cli", "examples/gpt2.yaml", "-i"])

    assert calls == [(["cli", "-i", "examples/gpt2.yaml"], "brainsurgery")]


def test_main_default_cli_reorders_options_after_config_items(monkeypatch) -> None:
    calls: list[tuple[list[str], str]] = []

    def _fake_app(*, args, prog_name):  # type: ignore[no-untyped-def]
        calls.append((list(args), prog_name))

    monkeypatch.setattr(brainsurgery, "app", _fake_app)

    brainsurgery.main(["examples/gpt2.yaml", "-i"])

    assert calls == [(["cli", "-i", "examples/gpt2.yaml"], "brainsurgery")]


def test_main_reorders_long_gpu_cache_options_after_config_items(monkeypatch) -> None:
    calls: list[tuple[list[str], str]] = []

    def _fake_app(*, args, prog_name):  # type: ignore[no-untyped-def]
        calls.append((list(args), prog_name))

    monkeypatch.setattr(brainsurgery, "app", _fake_app)

    brainsurgery.main(
        [
            "cli",
            "examples/gpt2_kv_infer_eat_my.yaml",
            "--gpu-cache-debug",
            "--gpu-cache-bytes",
            "10485760",
        ]
    )

    assert calls == [
        (
            [
                "cli",
                "--gpu-cache-debug",
                "--gpu-cache-bytes",
                "10485760",
                "examples/gpt2_kv_infer_eat_my.yaml",
            ],
            "brainsurgery",
        )
    ]
