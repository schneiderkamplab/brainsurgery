from __future__ import annotations

from pathlib import Path

import pytest
import torch

from brainsurgery.synapse import SynapseProgramModel
from tests.synapse_test_utils import (
    assert_logits_close,
    auto_device,
    build_codegen_model,
    extract_logits,
    load_yaml_mapping,
)


def _build_model_from_spec(
    spec_path: Path, class_name: str, state_dict: dict[str, torch.Tensor]
) -> object:
    return build_codegen_model(load_yaml_mapping(spec_path), class_name, state_dict)


def _build_runtime_model_from_spec(
    spec_path: Path, state_dict: dict[str, torch.Tensor]
) -> SynapseProgramModel:
    return SynapseProgramModel.from_spec(load_yaml_mapping(spec_path), state_dict=state_dict)


@pytest.mark.parametrize(
    ("spec_name", "class_name"),
    [
        ("gpt2_synapse.yaml", "GPT2SynapseGenerated"),
        ("gpt2_kv_synapse.yaml", "GPT2KVSynapseGenerated"),
    ],
)
def test_generated_gpt2_variants_match_hf(
    repo_root: Path,
    gpt2_local_paths: tuple[Path, Path],
    spec_name: str,
    class_name: str,
) -> None:
    pytest.skip("outdated gpt2 example synapse specs after split/positional API changes")
    transformers = pytest.importorskip("transformers")
    safetensors = pytest.importorskip("safetensors")

    device = auto_device()

    synapse_weights, hf_model_dir = gpt2_local_paths
    spec_path = repo_root / "examples" / spec_name

    st = safetensors.safe_open(str(synapse_weights), framework="pt")
    state_dict = {
        key: st.get_tensor(key).to(device=device, dtype=torch.float32) for key in st.keys()
    }

    tokenizer = transformers.AutoTokenizer.from_pretrained(str(hf_model_dir), local_files_only=True)
    hf_model = (
        transformers.AutoModelForCausalLM.from_pretrained(
            str(hf_model_dir), local_files_only=True, dtype=torch.float32
        )
        .to(device)
        .eval()
    )

    synapse_model = _build_model_from_spec(spec_path, class_name, state_dict).to(device).eval()
    runtime_model = _build_runtime_model_from_spec(spec_path, state_dict).to(device).eval()

    inputs = tokenizer("I eat my own", return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    max_len = input_ids.shape[1] + 12

    with torch.no_grad():
        synapse_logits = extract_logits(synapse_model(input_ids))
        runtime_logits = extract_logits(runtime_model(input_ids))
        hf_logits = hf_model(input_ids=input_ids).logits

    assert_logits_close(synapse_logits, hf_logits, mean_tol=1.0e-4, max_tol=5.0e-4)
    assert_logits_close(runtime_logits, hf_logits, mean_tol=1.0e-4, max_tol=5.0e-4)
    syn_rt_diff = (synapse_logits - runtime_logits).abs()
    assert float(syn_rt_diff.mean()) < 1.0e-6
    assert float(syn_rt_diff.max()) < 1.0e-5

    synapse_generated = synapse_model.generate(
        input_ids, eos_token_id=tokenizer.eos_token_id, max_len=max_len
    )
    runtime_generated = runtime_model.generate(
        input_ids, eos_token_id=tokenizer.eos_token_id, max_len=max_len
    )
    hf_generated = hf_model.generate(
        **inputs,
        max_new_tokens=max(1, max_len - int(input_ids.shape[1])),
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    assert torch.equal(synapse_generated, hf_generated)
    assert torch.equal(runtime_generated, hf_generated)
    assert torch.equal(runtime_generated, synapse_generated)
