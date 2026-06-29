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
    ensure_model_downloaded(repo_root=repo_root, config=pytestconfig, spec=MODEL_SPECS["gpt2"])
    synapse_weights = ensure_gpt2_weights_alias(repo_root, pytestconfig)
    hf_model_dir = repo_root / "models" / "gpt2"
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
def gemma4_e2b_local_path(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    return ensure_model_downloaded(
        repo_root=repo_root, config=pytestconfig, spec=MODEL_SPECS["gemma4_e2b"]
    )


@pytest.fixture(scope="session")
def gemma4_e4b_local_path(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    return ensure_model_downloaded(
        repo_root=repo_root, config=pytestconfig, spec=MODEL_SPECS["gemma4_e4b"]
    )


@pytest.fixture(scope="session")
def gemma4_26b_a4b_local_path(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    return ensure_model_downloaded(
        repo_root=repo_root, config=pytestconfig, spec=MODEL_SPECS["gemma4_26b_a4b"]
    )


@pytest.fixture(scope="session")
def gemma4_31b_local_path(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    return ensure_model_downloaded(
        repo_root=repo_root, config=pytestconfig, spec=MODEL_SPECS["gemma4_31b"]
    )


@pytest.fixture(scope="session")
def olmoe_local_path(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    return ensure_model_downloaded(
        repo_root=repo_root,
        config=pytestconfig,
        spec=MODEL_SPECS["olmoe_1b_7b_0924"],
    )


@pytest.fixture(scope="session")
def olmo3_1025_7b_local_path(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    return ensure_model_downloaded(
        repo_root=repo_root,
        config=pytestconfig,
        spec=MODEL_SPECS["olmo3_1025_7b"],
    )


@pytest.fixture(scope="session")
def olmo3_7b_instruct_local_path(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    return ensure_model_downloaded(
        repo_root=repo_root,
        config=pytestconfig,
        spec=MODEL_SPECS["olmo3_7b_instruct"],
    )


@pytest.fixture(scope="session")
def olmo3_7b_think_local_path(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    return ensure_model_downloaded(
        repo_root=repo_root,
        config=pytestconfig,
        spec=MODEL_SPECS["olmo3_7b_think"],
    )


@pytest.fixture(scope="session")
def olmo_2_1b_local_path(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    return ensure_model_downloaded(
        repo_root=repo_root,
        config=pytestconfig,
        spec=MODEL_SPECS["olmo_2_1b"],
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
def comma_local_path(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    return ensure_model_downloaded(
        repo_root=repo_root,
        config=pytestconfig,
        spec=MODEL_SPECS["comma"],
    )


@pytest.fixture(scope="session")
def dfm_decoder_local_path(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    return ensure_model_downloaded(
        repo_root=repo_root,
        config=pytestconfig,
        spec=MODEL_SPECS["dfm_decoder"],
    )


@pytest.fixture(scope="session")
def phi3_mini_4k_instruct_local_path(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    return ensure_model_downloaded(
        repo_root=repo_root,
        config=pytestconfig,
        spec=MODEL_SPECS["phi3_mini_4k_instruct"],
    )


@pytest.fixture(scope="session")
def smollm_135m_local_path(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    return ensure_model_downloaded(
        repo_root=repo_root,
        config=pytestconfig,
        spec=MODEL_SPECS["smollm_135m"],
    )


@pytest.fixture(scope="session")
def smollm_360m_local_path(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    return ensure_model_downloaded(
        repo_root=repo_root,
        config=pytestconfig,
        spec=MODEL_SPECS["smollm_360m"],
    )


@pytest.fixture(scope="session")
def smollm_1_7b_local_path(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    return ensure_model_downloaded(
        repo_root=repo_root,
        config=pytestconfig,
        spec=MODEL_SPECS["smollm_1_7b"],
    )


@pytest.fixture(scope="session")
def smollm2_135m_local_path(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    return ensure_model_downloaded(
        repo_root=repo_root,
        config=pytestconfig,
        spec=MODEL_SPECS["smollm2_135m"],
    )


@pytest.fixture(scope="session")
def smollm2_360m_local_path(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    return ensure_model_downloaded(
        repo_root=repo_root,
        config=pytestconfig,
        spec=MODEL_SPECS["smollm2_360m"],
    )


@pytest.fixture(scope="session")
def smollm2_1_7b_local_path(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    return ensure_model_downloaded(
        repo_root=repo_root,
        config=pytestconfig,
        spec=MODEL_SPECS["smollm2_1_7b"],
    )


@pytest.fixture(scope="session")
def smollm3_3b_local_path(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    return ensure_model_downloaded(
        repo_root=repo_root,
        config=pytestconfig,
        spec=MODEL_SPECS["smollm3_3b"],
    )


@pytest.fixture(scope="session")
def smollm3_3b_base_local_path(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    return ensure_model_downloaded(
        repo_root=repo_root,
        config=pytestconfig,
        spec=MODEL_SPECS["smollm3_3b_base"],
    )


@pytest.fixture(scope="session")
def black_mamba_local_path(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    return ensure_model_downloaded(
        repo_root=repo_root,
        config=pytestconfig,
        spec=MODEL_SPECS["black_mamba"],
    )


@pytest.fixture(scope="session")
def apertus_8b_local_path(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    return ensure_model_downloaded(
        repo_root=repo_root,
        config=pytestconfig,
        spec=MODEL_SPECS["apertus_8b"],
    )


@pytest.fixture(scope="session")
def bert_local_path(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    return ensure_model_downloaded(
        repo_root=repo_root,
        config=pytestconfig,
        spec=MODEL_SPECS["bert"],
    )


@pytest.fixture(scope="session")
def roberta_local_path(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    return ensure_model_downloaded(
        repo_root=repo_root,
        config=pytestconfig,
        spec=MODEL_SPECS["roberta"],
    )


@pytest.fixture(scope="session")
def camembert_local_path(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    return ensure_model_downloaded(
        repo_root=repo_root,
        config=pytestconfig,
        spec=MODEL_SPECS["camembert"],
    )


@pytest.fixture(scope="session")
def xlm_roberta_local_path(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    return ensure_model_downloaded(
        repo_root=repo_root,
        config=pytestconfig,
        spec=MODEL_SPECS["xlm_roberta"],
    )


@pytest.fixture(scope="session")
def distilbert_local_path(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    return ensure_model_downloaded(
        repo_root=repo_root,
        config=pytestconfig,
        spec=MODEL_SPECS["distilbert"],
    )


@pytest.fixture(scope="session")
def electra_local_path(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    return ensure_model_downloaded(
        repo_root=repo_root,
        config=pytestconfig,
        spec=MODEL_SPECS["electra"],
    )


@pytest.fixture(scope="session")
def albert_local_path(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    return ensure_model_downloaded(
        repo_root=repo_root,
        config=pytestconfig,
        spec=MODEL_SPECS["albert"],
    )


@pytest.fixture(scope="session")
def deberta_v2_local_path(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    return ensure_model_downloaded(
        repo_root=repo_root,
        config=pytestconfig,
        spec=MODEL_SPECS["deberta_v2"],
    )


@pytest.fixture(scope="session")
def longformer_local_path(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    return ensure_model_downloaded(
        repo_root=repo_root,
        config=pytestconfig,
        spec=MODEL_SPECS["longformer"],
    )


@pytest.fixture(scope="session")
def modernbert_local_path(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    return ensure_model_downloaded(
        repo_root=repo_root,
        config=pytestconfig,
        spec=MODEL_SPECS["modernbert"],
    )


@pytest.fixture(scope="session")
def t5_small_local_path(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    return ensure_model_downloaded(
        repo_root=repo_root,
        config=pytestconfig,
        spec=MODEL_SPECS["t5_small"],
    )


@pytest.fixture(scope="session")
def mt5_small_local_path(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    return ensure_model_downloaded(
        repo_root=repo_root,
        config=pytestconfig,
        spec=MODEL_SPECS["mt5_small"],
    )


@pytest.fixture(scope="session")
def bart_base_local_path(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    return ensure_model_downloaded(
        repo_root=repo_root,
        config=pytestconfig,
        spec=MODEL_SPECS["bart_base"],
    )


@pytest.fixture(scope="session")
def mbart_large_50_m2m_local_path(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    return ensure_model_downloaded(
        repo_root=repo_root,
        config=pytestconfig,
        spec=MODEL_SPECS["mbart_large_50_m2m"],
    )


@pytest.fixture(scope="session")
def marian_en_de_local_path(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    return ensure_model_downloaded(
        repo_root=repo_root,
        config=pytestconfig,
        spec=MODEL_SPECS["marian_en_de"],
    )


@pytest.fixture(scope="session")
def t5gemma_s_s_ul2_local_path(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    return ensure_model_downloaded(
        repo_root=repo_root,
        config=pytestconfig,
        spec=MODEL_SPECS["t5gemma_s_s_ul2"],
    )


@pytest.fixture(scope="session")
def t5gemma2_270m_local_path(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    return ensure_model_downloaded(
        repo_root=repo_root,
        config=pytestconfig,
        spec=MODEL_SPECS["t5gemma2_270m"],
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
