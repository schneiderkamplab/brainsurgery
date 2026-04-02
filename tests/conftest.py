from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from tests.model_downloads import (
    MODEL_SPECS,
    ensure_gpt2_weights_alias,
    ensure_matrix_models,
    ensure_model_downloaded,
)


class SingleModelProvider:
    def __init__(self, state_dict: object, model: str = "model") -> None:
        self.state_dict = state_dict
        self._model = model

    def get_state_dict(self, model: str):
        assert model == self._model
        return self.state_dict


class MultiModelProvider:
    def __init__(self, state_dicts: dict[str, dict[str, torch.Tensor]]) -> None:
        self.state_dicts = state_dicts

    def get_state_dict(self, model: str):
        return self.state_dicts[model]


@pytest.fixture
def single_model_provider() -> Callable[[object, str], SingleModelProvider]:
    def _make(state_dict: object, model: str = "model") -> SingleModelProvider:
        return SingleModelProvider(state_dict=state_dict, model=model)

    return _make


@pytest.fixture
def multi_model_provider() -> Callable[[dict[str, dict[str, torch.Tensor]]], MultiModelProvider]:
    def _make(state_dicts: dict[str, dict[str, torch.Tensor]]) -> MultiModelProvider:
        return MultiModelProvider(state_dicts=state_dicts)

    return _make


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def gpt2_local_paths(repo_root: Path, pytestconfig: pytest.Config) -> tuple[Path, Path]:
    ensure_model_downloaded(repo_root=repo_root, config=pytestconfig, spec=MODEL_SPECS["gpt2.old"])
    synapse_weights = ensure_gpt2_weights_alias(repo_root, pytestconfig)
    hf_model_dir = repo_root / "models" / "gpt2.old"
    return synapse_weights, hf_model_dir


@pytest.fixture(scope="session")
def gemma3_local_path(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    return ensure_model_downloaded(
        repo_root=repo_root, config=pytestconfig, spec=MODEL_SPECS["gemma3"]
    )


@pytest.fixture(scope="session")
def gemma3_1b_local_path(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    return ensure_model_downloaded(
        repo_root=repo_root, config=pytestconfig, spec=MODEL_SPECS["gemma3_1b"]
    )


@pytest.fixture(scope="session")
def gemma3_4b_local_path(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    return ensure_model_downloaded(
        repo_root=repo_root, config=pytestconfig, spec=MODEL_SPECS["gemma3_4b"]
    )


@pytest.fixture(scope="session")
def olmoe_local_path(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    return ensure_model_downloaded(
        repo_root=repo_root,
        config=pytestconfig,
        spec=MODEL_SPECS["olmoe_1b_7b_0924"],
    )


@pytest.fixture(scope="session")
def glm4_5_air_local_path(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    return ensure_model_downloaded(
        repo_root=repo_root,
        config=pytestconfig,
        spec=MODEL_SPECS["glm_4_5_air"],
    )


@pytest.fixture(scope="session")
def deepseek_v2_lite_local_path(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    return ensure_model_downloaded(
        repo_root=repo_root,
        config=pytestconfig,
        spec=MODEL_SPECS["deepseek_v2_lite"],
    )


@pytest.fixture(scope="session")
def black_mamba_local_path(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    return ensure_model_downloaded(
        repo_root=repo_root,
        config=pytestconfig,
        spec=MODEL_SPECS["black_mamba"],
    )


@pytest.fixture(scope="session")
def nemotron3_local_path(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    return ensure_model_downloaded(
        repo_root=repo_root,
        config=pytestconfig,
        spec=MODEL_SPECS["nemotron3"],
    )


@pytest.fixture(scope="session")
def ensure_matrix_test_models(repo_root: Path, pytestconfig: pytest.Config) -> None:
    ensure_matrix_models(repo_root, pytestconfig)
