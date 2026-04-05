from __future__ import annotations

from pathlib import Path
from typing import Any

from brainsurgery.synapse import lower_axon_program_to_synapse_spec, parse_axon_program_from_path
from tests.model_downloads import MATRIX_AXON_MODEL_DIR_PAIRS, MODEL_SPECS


def _load_axon_spec(path: Path) -> dict[str, Any]:
    modules = parse_axon_program_from_path(path)
    return lower_axon_program_to_synapse_spec(modules)


def test_deepseek_v2_lite_is_registered_in_download_specs_and_matrix() -> None:
    assert MODEL_SPECS["deepseek_v2_lite"].repo_id == "deepseek-ai/DeepSeek-V2-Lite"
    assert MODEL_SPECS["deepseek_v2_lite"].local_dir == "deepseek_v2_lite"
    assert ("deepseek_v2_lite", "deepseek_v2_lite") in MATRIX_AXON_MODEL_DIR_PAIRS


def test_deepseek_v2_lite_axon_lowers_with_expected_symbols(repo_root: Path) -> None:
    spec = _load_axon_spec(repo_root / "examples" / "deepseek_v2_lite.axon")

    assert spec.get("synapse") == 1
    model = spec.get("model", {})
    assert model.get("outputs") == {"logits": "logits", "new_kv": "new_kv"}

    symbols = model.get("symbols", {})
    assert symbols.get("D") == 2048
    assert symbols.get("V") == 102400
    assert symbols.get("L") == 27
    assert symbols.get("H") == 16
    assert symbols.get("QKNOPE") == 128
    assert symbols.get("QKROPE") == 64
    assert symbols.get("VHD") == 128
    assert symbols.get("QHD") in (192, "QKNOPE + QKROPE", "QKNOPE+QKROPE")
    assert symbols.get("KVR") == 512
    assert symbols.get("KVPROJ") in (4096, "H * (QKNOPE + VHD)", "H*(QKNOPE+VHD)")
    assert symbols.get("FFN") == 10944
    assert symbols.get("E") == 64
    assert symbols.get("EPT") == 6
    assert symbols.get("EM") == 1408
    assert symbols.get("SE") == 2
    assert symbols.get("EPS") == 1.0e-06
    assert symbols.get("THETA") == 10000.0
    assert symbols.get("C") == 163840
    assert symbols.get("FIRST_DENSE") == 1

    blocks = model.get("blocks", {})
    assert "deepseek_v2_lite_attn" in blocks
    assert "deepseek_v2_lite_dense_block" in blocks
    assert "deepseek_v2_lite_moe_block" in blocks


def test_deepseek_v2_lite_nested_path_block_calls_keep_relative_param_paths(repo_root: Path) -> None:
    modules = parse_axon_program_from_path(repo_root / "examples" / "deepseek_v2_lite.axon")
    spec = lower_axon_program_to_synapse_spec(
        modules, main_module="deepseek_v2_lite_layer0_input_norm_stage"
    )

    graph = spec["model"]["graph"]
    loop_node = next(node for item in graph for node in item.values() if node.get("_op") == "for")
    call_node = loop_node["_body"][0]["n_call_28"]
    assert call_node["_target"] == "rms__path_input_layernorm"

    block = spec["model"]["blocks"]["rms__path_input_layernorm"]
    rms_node = block["graph"][-1]["n_op_27"]
    assert rms_node["_params"]["weight"] == "input_layernorm.weight"


def test_deepseek_v2_lite_nested_self_attn_path_calls_keep_runtime_scope(repo_root: Path) -> None:
    modules = parse_axon_program_from_path(repo_root / "examples" / "deepseek_v2_lite.axon")
    spec = lower_axon_program_to_synapse_spec(modules, main_module="deepseek_v2_lite")

    attn_block = spec["model"]["blocks"]["deepseek_v2_lite_attn"]
    call_node = next(
        node
        for item in attn_block["graph"]
        for node in item.values()
        if node.get("_op") == "call" and node.get("_target") == "rms__path_kv_a_layernorm"
    )
    assert call_node["_scope"] == "self_attn"

    block = spec["model"]["blocks"]["rms__path_kv_a_layernorm"]
    rms_node = block["graph"][-1]["n_op_27"]
    assert rms_node["_params"]["weight"] == "kv_a_layernorm.weight"


def test_deepseek_v2_lite_routed_expert_paths_lower_relative_to_expert_scope(
    repo_root: Path,
) -> None:
    modules = parse_axon_program_from_path(repo_root / "examples" / "deepseek_v2_lite.axon")
    spec = lower_axon_program_to_synapse_spec(modules, main_module="deepseek_v2_lite")

    moe_block = spec["model"]["blocks"]["deepseek_v2_lite_moe_block"]
    expert_loop = next(
        node
        for item in moe_block["graph"]
        for node in item.values()
        if node.get("_op") == "for" and node.get("_scope") == "mlp.experts"
    )
    linear_nodes = [
        node
        for item in expert_loop["_body"]
        for node in item.values()
        if node.get("_op") == "linear"
    ]
    weights = [node["_params"]["weight"] for node in linear_nodes]
    scopes = [node.get("_scope") for node in linear_nodes]

    assert weights == ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]
    assert scopes == [None, None, None]
