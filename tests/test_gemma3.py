from __future__ import annotations

from pathlib import Path

import pytest
import torch

from brainsurgery.synapse import SynapseProgramModel
from tests.synapse_test_utils import (
    assert_masked_logits_close,
    auto_device,
    build_codegen_model,
    extract_logits,
    load_yaml_mapping,
)


def _build_runtime_model_from_spec(
    spec_path: Path, state_dict: dict[str, torch.Tensor]
) -> SynapseProgramModel:
    return SynapseProgramModel.from_spec(load_yaml_mapping(spec_path), state_dict=state_dict)


@pytest.mark.parametrize(
    ("texts",),
    [
        (["I eat my own"],),
        (["The future of AI is", "Hello world"],),
    ],
)
def test_generated_gemma3_matches_hf(
    repo_root: Path, gemma3_local_path: Path, texts: list[str]
) -> None:
    pytest.skip("outdated gemma3 synapse spec uses legacy repeat argument form")
    transformers = pytest.importorskip("transformers")
    safetensors = pytest.importorskip("safetensors")
    device = auto_device()

    spec_path = repo_root / "examples" / "gemma3_270m_synapse.yaml"
    weights_path = gemma3_local_path / "model.safetensors"
    if not weights_path.exists():
        pytest.skip(f"missing Gemma3 checkpoint: {weights_path}")

    st = safetensors.safe_open(str(weights_path), framework="pt")
    state_dict = {
        key: st.get_tensor(key).to(device=device, dtype=torch.float32) for key in st.keys()
    }

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(gemma3_local_path), local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    hf_model = (
        transformers.AutoModelForCausalLM.from_pretrained(
            str(gemma3_local_path), local_files_only=True, dtype=torch.float32
        )
        .to(device)
        .eval()
    )

    # Match deterministic greedy decoding behavior used by the synapse generated model.
    hf_model.generation_config.do_sample = False
    hf_model.generation_config.top_p = None
    hf_model.generation_config.top_k = None

    synapse_model = (
        build_codegen_model(load_yaml_mapping(spec_path), "Gemma3Generated", state_dict)
        .to(device)
        .eval()
    )
    runtime_model = _build_runtime_model_from_spec(spec_path, state_dict).to(device).eval()

    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    assert attention_mask is not None
    max_len = input_ids.shape[1] + 12

    with torch.no_grad():
        synapse_logits = extract_logits(synapse_model(input_ids, attn_mask=attention_mask))
        runtime_logits = extract_logits(runtime_model(input_ids, attn_mask=attention_mask))
        hf_logits = hf_model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).logits

    assert_masked_logits_close(
        synapse_logits,
        hf_logits,
        attention_mask,
        mean_tol=1.0e-4,
        max_tol=5.0e-4,
    )
    assert_masked_logits_close(
        runtime_logits,
        hf_logits,
        attention_mask,
        mean_tol=1.0e-4,
        max_tol=5.0e-4,
    )
    assert_masked_logits_close(
        synapse_logits,
        runtime_logits,
        attention_mask,
        mean_tol=1.0e-6,
        max_tol=1.0e-5,
    )

    synapse_generated = synapse_model.generate(
        input_ids,
        attn_mask=attention_mask,
        eos_token_id=tokenizer.eos_token_id,
        max_len=max_len,
    )
    runtime_generated = runtime_model.generate(
        input_ids,
        attn_mask=attention_mask,
        eos_token_id=tokenizer.eos_token_id,
        max_len=max_len,
    )
    hf_generated = hf_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max(1, max_len - int(input_ids.shape[1])),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    assert torch.equal(synapse_generated, hf_generated)
    assert torch.equal(runtime_generated, hf_generated)
    assert torch.equal(runtime_generated, synapse_generated)
