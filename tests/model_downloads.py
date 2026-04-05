from __future__ import annotations

from pathlib import Path

import pytest

from brainsurgery.synapse import matrix_models as _matrix_models

MATRIX_AXON_MODEL_DIRS = _matrix_models.MATRIX_AXON_MODEL_DIRS
MATRIX_AXON_MODEL_DIR_PAIRS = _matrix_models.MATRIX_AXON_MODEL_DIR_PAIRS
MODEL_SPECS = _matrix_models.MODEL_SPECS
ModelDownloadSpec = _matrix_models.ModelDownloadSpec

__all__ = [
    "MATRIX_AXON_MODEL_DIR_PAIRS",
    "MATRIX_AXON_MODEL_DIRS",
    "MODEL_SPECS",
    "ModelDownloadSpec",
    "ensure_gpt2_weights_alias",
    "ensure_matrix_models",
    "ensure_model_downloaded",
]


def _status(config: pytest.Config, message: str) -> None:
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(f"[model-download] {message}")
    else:
        print(f"[model-download] {message}", flush=True)


def ensure_model_downloaded(
    *,
    repo_root: Path,
    config: pytest.Config,
    spec: ModelDownloadSpec,
) -> Path:
    return _matrix_models.ensure_model_downloaded(
        repo_root=repo_root,
        spec=spec,
        status_cb=lambda message: _status(config, message),
    )


def ensure_gpt2_weights_alias(repo_root: Path, config: pytest.Config) -> Path:
    _ = config
    return _matrix_models.ensure_gpt2_weights_alias(repo_root)


def ensure_matrix_models(repo_root: Path, config: pytest.Config) -> None:
    _matrix_models.ensure_matrix_models(
        repo_root=repo_root,
        status_cb=lambda message: _status(config, message),
    )
