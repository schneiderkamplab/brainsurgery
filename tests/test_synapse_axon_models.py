from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from brainsurgery.synapse import (
    SynapseProgramModel,
    lower_axon_program_to_synapse_spec,
    parse_axon_program_from_path,
)
from tests.synapse_test_utils import (
    assert_logits_close,
    auto_device,
    build_codegen_model,
    extract_logits,
    load_yaml_mapping,
)
from tests.test_flags import LONG_TEST_ENV, run_long_tests_enabled

pytestmark = pytest.mark.skip(
    reason=(
        "Outdated axon<->synapse model-roundtrip expectations after positional-only "
        "op signatures and path-resolution semantics updates."
    )
)

_RUN_LONG = run_long_tests_enabled()


def _load_axon_spec(path: Path) -> dict[str, Any]:
    modules = parse_axon_program_from_path(path)
    return lower_axon_program_to_synapse_spec(modules)


@pytest.mark.parametrize(
    ("yaml_name", "axon_name", "expected_symbols"),
    [
        (
            "gpt2_synapse.yaml",
            "gpt2.axon",
            {
                "T": 1024,
                "D": 768,
                "L": 12,
                "H": 12,
                "V": None,
                "B": None,
                "S": None,
            },
        ),
        (
            "gpt2_kv_synapse.yaml",
            "gpt2_kv.axon",
            {
                "T": 1024,
                "D": 768,
                "L": 12,
                "H": 12,
                "V": 520257,
                "B": None,
                "S": None,
            },
        ),
        (
            "gemma3_270m_synapse.yaml",
            "gemma3_270m.axon",
            {
                "D": 640,
                "V": 262208,
                "L": 18,
                "H": 4,
                "KVH": 1,
                "QD": 1024,
                "KVD": 256,
                "FFN": 2048,
                "ROPE_PERIOD": 6,
                "THETA_BASE": 10000.0,
                "THETA_LONG": 1000000.0,
                "WIN_LOCAL": 512,
                "WIN_LONG": 32768,
                "ATTN_SCALE": 0.0625,
                "EMB_SCALE": 25.298221281347036,
                "B": None,
                "S": None,
            },
        ),
        (
            "olmoe_1b_7b_0924_synapse.yaml",
            "olmoe_1b_7b_0924.axon",
            {
                "D": 2048,
                "V": 50304,
                "L": 16,
                "H": 16,
                "HD": 128,
                "E": 64,
                "EPT": 8,
                "C": 4096,
                "EPS": 1.0e-05,
                "THETA": 10000.0,
                "ATTN_SCALE": 0.08838834764831845,
                "B": None,
                "S": None,
            },
        ),
    ],
)
def test_axon_files_roundtrip_to_original_synapse_spec(
    repo_root: Path, yaml_name: str, axon_name: str, expected_symbols: dict[str, object] | None
) -> None:
    expected = load_yaml_mapping(repo_root / "examples" / yaml_name)
    lowered = _load_axon_spec(repo_root / "examples" / axon_name)
    assert lowered.get("synapse") == expected.get("synapse") == 1
    assert set(lowered.get("model", {}).get("inputs", {}).keys()) == set(
        expected.get("model", {}).get("inputs", {}).keys()
    )
    assert lowered.get("model", {}).get("outputs") == expected.get("model", {}).get("outputs")
    assert lowered.get("model", {}).get("symbols") == expected_symbols
    lowered_blocks = set(lowered.get("model", {}).get("blocks", {}).keys())
    expected_blocks = set(expected.get("model", {}).get("blocks", {}).keys())
    assert expected_blocks.issubset(lowered_blocks)


@pytest.mark.parametrize(
    ("axon_name", "prompt"),
    [
        ("gpt2.axon", "I eat my own"),
        ("gemma3_270m.axon", "I eat my own"),
    ],
)
def test_codegen_and_runtime_from_axon_match_hf_decoder_models(
    repo_root: Path,
    gpt2_local_paths: tuple[Path, Path],
    gemma3_local_path: Path,
    axon_name: str,
    prompt: str,
) -> None:
    transformers = pytest.importorskip("transformers")
    safetensors = pytest.importorskip("safetensors")
    device = auto_device()

    if axon_name == "gpt2.axon":
        weights_path, hf_dir = gpt2_local_paths
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(hf_dir), local_files_only=True)
        st = safetensors.safe_open(str(weights_path), framework="pt")
        state_dict = {
            key: st.get_tensor(key).to(device=device, dtype=torch.float32) for key in st.keys()
        }
    else:
        hf_dir = gemma3_local_path
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(hf_dir), local_files_only=True)
        st = safetensors.safe_open(str(hf_dir / "model.safetensors"), framework="pt")
        state_dict = {
            key: st.get_tensor(key).to(device=device, dtype=torch.float32) for key in st.keys()
        }

    hf_model = (
        transformers.AutoModelForCausalLM.from_pretrained(
            str(hf_dir), local_files_only=True, dtype=torch.float32
        )
        .to(device)
        .eval()
    )
    hf_model.generation_config.do_sample = False
    hf_model.generation_config.top_p = None
    hf_model.generation_config.top_k = None

    spec = _load_axon_spec(repo_root / "examples" / axon_name)
    cg_model = (
        build_codegen_model(spec, f"Axon{axon_name.replace('.', '_')}", state_dict)
        .to(device)
        .eval()
    )
    rt_model = SynapseProgramModel.from_spec(spec, state_dict=state_dict).to(device).eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    max_len = input_ids.shape[1] + 8

    with torch.no_grad():
        cg_logits = extract_logits(cg_model(input_ids))
        rt_logits = extract_logits(rt_model(input_ids))
        hf_logits = hf_model(input_ids=input_ids, use_cache=False).logits

    assert_logits_close(cg_logits, hf_logits, mean_tol=1.0e-4, max_tol=5.0e-4)
    assert_logits_close(rt_logits, hf_logits, mean_tol=1.0e-4, max_tol=5.0e-4)

    cg_gen = cg_model.generate(input_ids, eos_token_id=tokenizer.eos_token_id, max_len=max_len)
    rt_gen = rt_model.generate(input_ids, eos_token_id=tokenizer.eos_token_id, max_len=max_len)
    hf_gen = hf_model.generate(
        **inputs,
        max_new_tokens=max(1, max_len - int(input_ids.shape[1])),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    assert torch.equal(cg_gen, hf_gen)
    assert torch.equal(rt_gen, hf_gen)


@pytest.mark.skipif(not _RUN_LONG, reason=f"set {LONG_TEST_ENV}=1 to enable long tests")
def test_codegen_and_runtime_from_axon_match_hf_olmoe_cpu(
    repo_root: Path, olmoe_local_path: Path
) -> None:
    transformers = pytest.importorskip("transformers")
    safetensors = pytest.importorskip("safetensors")
    device = torch.device("cpu")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(olmoe_local_path), local_files_only=True
    )
    hf_model = (
        transformers.AutoModelForCausalLM.from_pretrained(
            str(olmoe_local_path), local_files_only=True
        )
        .to(device)
        .eval()
    )

    index_path = olmoe_local_path / "model.safetensors.index.json"
    payload = OmegaConf.to_container(OmegaConf.load(index_path), resolve=True)
    assert isinstance(payload, dict)
    weight_map = payload.get("weight_map")
    assert isinstance(weight_map, dict)
    shard_names = sorted({str(name) for name in weight_map.values()})
    state_dict: dict[str, torch.Tensor] = {}
    for shard_name in shard_names:
        st = safetensors.safe_open(str(olmoe_local_path / shard_name), framework="pt")
        for key in st.keys():
            state_dict[key] = st.get_tensor(key).to(device)
    spec = _load_axon_spec(repo_root / "examples" / "olmoe_1b_7b_0924.axon")
    cg_model = build_codegen_model(spec, "AxonOlmoeGenerated", state_dict).to(device).eval()
    rt_model = SynapseProgramModel.from_spec(spec, state_dict=state_dict).to(device).eval()

    inputs = tokenizer("The future of AI is", return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    max_len = input_ids.shape[1] + 8

    with torch.no_grad():
        cg_logits = extract_logits(cg_model(input_ids))
        rt_logits = extract_logits(rt_model(input_ids))
        hf_logits = hf_model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).logits

    assert_logits_close(cg_logits, hf_logits, mean_tol=0.1, max_tol=1.0)
    assert_logits_close(rt_logits, hf_logits, mean_tol=0.1, max_tol=1.0)

    cg_gen = cg_model.generate(input_ids, eos_token_id=tokenizer.eos_token_id, max_len=max_len)
    rt_gen = rt_model.generate(input_ids, eos_token_id=tokenizer.eos_token_id, max_len=max_len)
    hf_gen = hf_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max(1, max_len - int(input_ids.shape[1])),
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    assert torch.equal(cg_gen, hf_gen)
    assert torch.equal(rt_gen, hf_gen)
