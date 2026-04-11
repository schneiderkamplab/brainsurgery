from __future__ import annotations

from tests.model_downloads import MATRIX_AXON_MODEL_DIR_PAIRS, MODEL_SPECS


def test_deepseek_v2_lite_is_registered_in_download_specs_and_matrix() -> None:
    assert MODEL_SPECS["deepseek_v2_lite"].repo_id == "deepseek-ai/DeepSeek-V2-Lite"
    assert MODEL_SPECS["deepseek_v2_lite"].local_dir == "deepseek_v2_lite"
    assert ("deepseek_v2_lite", "deepseek_v2_lite") in MATRIX_AXON_MODEL_DIR_PAIRS
