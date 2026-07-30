"""Test vLLM codegen changes across architectures."""
import sys
import os
import json
import traceback

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
os.environ.setdefault("HF_TOKEN", os.environ.get("HF_TOKEN", ""))

REPO = "/work/dfm/jacobwashere/brainsurgery"
sys.path.insert(0, REPO)

from brainsurgery.synapse.axon_test import run_axon_test
from pathlib import Path

MODELS_DIR = Path(REPO) / "models"
AXON_DIR = Path(REPO) / "brainsurgery/synapse/models"

TESTS = [
    # Existing tests
    (AXON_DIR / "smollm/SmolLM-135M.axon", MODELS_DIR / "HuggingFaceTB/SmolLM-135M", "SmolLM-135M"),
    (AXON_DIR / "smollm/SmolLM2-135M.axon", MODELS_DIR / "HuggingFaceTB/SmolLM2-135M", "SmolLM2-135M"),
    (AXON_DIR / "smollm/SmolLM2-360M.axon", MODELS_DIR / "HuggingFaceTB/SmolLM2-360M", "SmolLM2-360M"),
    (AXON_DIR / "smollm/SmolLM2-1.7B.axon", MODELS_DIR / "HuggingFaceTB/SmolLM2-1.7B", "SmolLM2-1.7B"),
    (AXON_DIR / "smollm3/SmolLM3-3B.axon", MODELS_DIR / "HuggingFaceTB/SmolLM3-3B", "SmolLM3-3B"),
    (AXON_DIR / "starcoder2/starcoder2-3b.axon", MODELS_DIR / "bigcode/starcoder2-3b", "starcoder2-3b"),
    (AXON_DIR / "granite/granite-3.3-2b-base.axon", MODELS_DIR / "ibm-granite/granite-3.3-2b-base", "granite-3.3-2b-base"),
    (AXON_DIR / "gemma2/gemma-2-2b.axon", MODELS_DIR / "google/gemma-2-2b", "gemma-2-2b"),
    (AXON_DIR / "gemma3/gemma-3-4b.axon", MODELS_DIR / "google/gemma-3-4b-pt", "gemma-3-4b"),
    (AXON_DIR / "qwen2/Qwen2.5-0.5B.axon", MODELS_DIR / "Qwen/Qwen2.5-0.5B", "Qwen2.5-0.5B"),
    (AXON_DIR / "phi4/Phi-4-mini-instruct.axon", MODELS_DIR / "microsoft/Phi-4-mini-instruct", "Phi-4-mini-instruct"),
    (AXON_DIR / "falcon/falcon_rw_1b.axon", MODELS_DIR / "tiiuae/falcon-rw-1b", "falcon-rw-1b"),
    # New models - diverse architectures
    (AXON_DIR / "gpt2/gpt2.axon", MODELS_DIR / "openai-community/gpt2", "gpt2"),
    (AXON_DIR / "bloom/bloom-560m.axon", MODELS_DIR / "bigscience/bloom-560m", "bloom-560m"),
    (AXON_DIR / "bloom/bloom-1b1.axon", MODELS_DIR / "bigscience/bloom-1b1", "bloom-1b1"),
    (AXON_DIR / "bloom/bloom-3b.axon", MODELS_DIR / "bigscience/bloom-3b", "bloom-3b"),
    (AXON_DIR / "smollm/SmolLM-360M.axon", MODELS_DIR / "HuggingFaceTB/SmolLM-360M", "SmolLM-360M"),
    (AXON_DIR / "smollm/SmolLM-1.7B.axon", MODELS_DIR / "HuggingFaceTB/SmolLM-1.7B", "SmolLM-1.7B"),
    (AXON_DIR / "gemma3/gemma-3-270m.axon", MODELS_DIR / "google/gemma-3-270m-it", "gemma-3-270m"),
    (AXON_DIR / "gemma3/gemma-3-1b.axon", MODELS_DIR / "google/gemma-3-1b-it", "gemma-3-1b"),
    (AXON_DIR / "pleias/Pleias-Pico.axon", MODELS_DIR / "PleIAs/Pleias-Pico", "Pleias-Pico"),
    (AXON_DIR / "pleias/Pleias-Nano.axon", MODELS_DIR / "PleIAs/Pleias-Nano", "Pleias-Nano"),
    (AXON_DIR / "exaone4/EXAONE-4.0-1.2B.axon", MODELS_DIR / "LGAI-EXAONE/EXAONE-4.0-1.2B", "EXAONE-4.0-1.2B"),
    (AXON_DIR / "olmo2/OLMo-2-0425-1B.axon", MODELS_DIR / "allenai/OLMo-2-0425-1B", "OLMo-2-0425-1B"),
    (AXON_DIR / "granite/granite-3.1-2b-instruct.axon", MODELS_DIR / "ibm-granite/granite-3.1-2b-instruct", "granite-3.1-2b-instruct"),
    (AXON_DIR / "phi3minimedium/Phi-3-mini-4k-instruct.axon", MODELS_DIR / "microsoft/Phi-3-mini-4k-instruct", "Phi-3-mini-4k"),
    (AXON_DIR / "glm/glm-edge-1.5b-chat.axon", MODELS_DIR / "zai-org/glm-edge-1.5b-chat", "glm-edge-1.5b"),
    (AXON_DIR / "gemma/gemma-2b.axon", MODELS_DIR / "google/gemma-2b", "gemma-2b"),
    (AXON_DIR / "mistral/Mistral-7B-v0.1.axon", MODELS_DIR / "mistralai/Mistral-7B-v0.1", "Mistral-7B-v0.1"),
    (AXON_DIR / "llama3/Meta-Llama-3-8B.axon", MODELS_DIR / "meta-llama/Meta-Llama-3-8B", "Meta-Llama-3-8B"),
    (AXON_DIR / "gemma/gemma-7b.axon", MODELS_DIR / "google/gemma-7b", "gemma-7b"),
    (AXON_DIR / "olmo2/OLMo-2-1124-7B.axon", MODELS_DIR / "allenai/OLMo-2-1124-7B", "OLMo-2-1124-7B"),
    (AXON_DIR / "deepseekv2/DeepSeek-V2-Lite.axon", MODELS_DIR / "deepseek-ai/DeepSeek-V2-Lite", "DeepSeek-V2-Lite"),
    (AXON_DIR / "stablelm/generic-stablelm.axon", MODELS_DIR / "stabilityai/stablelm-2-1_6b", "stablelm-2-1.6b"),
    # --- Expanded: same family as passing models ---
    (AXON_DIR / "bloom/bloom-1b7.axon", MODELS_DIR / "bigscience/bloom-1b7", "bloom-1b7"),
    (AXON_DIR / "bloom/bloom-7b1.axon", MODELS_DIR / "bigscience/bloom-7b1", "bloom-7b1"),
    (AXON_DIR / "granite/granite-3.1-2b-base.axon", MODELS_DIR / "ibm-granite/granite-3.1-2b-base", "granite-3.1-2b-base"),
    (AXON_DIR / "granite/granite-3.3-2b-instruct.axon", MODELS_DIR / "ibm-granite/granite-3.3-2b-instruct", "granite-3.3-2b-instruct"),
    (AXON_DIR / "granite/granite-3.1-8b-base.axon", MODELS_DIR / "ibm-granite/granite-3.1-8b-base", "granite-3.1-8b-base"),
    (AXON_DIR / "granite/granite-3.3-8b-instruct.axon", MODELS_DIR / "ibm-granite/granite-3.3-8b-instruct", "granite-3.3-8b-instruct"),
    (AXON_DIR / "qwen2/Qwen2.5-14B.axon", MODELS_DIR / "Qwen/Qwen2.5-14B", "Qwen2.5-14B"),
    (AXON_DIR / "qwen2/Qwen2.5-Coder-14B.axon", MODELS_DIR / "Qwen/Qwen2.5-Coder-14B", "Qwen2.5-Coder-14B"),
    (AXON_DIR / "mistral/Mistral-7B-v0.3.axon", MODELS_DIR / "mistralai/Mistral-7B-v0.3", "Mistral-7B-v0.3"),
    (AXON_DIR / "mistral/Mistral-7B-Instruct-v0.2.axon", MODELS_DIR / "mistralai/Mistral-7B-Instruct-v0.2", "Mistral-7B-Instruct-v0.2"),
    (AXON_DIR / "mistral/Mistral-Nemo-Base-2407.axon", MODELS_DIR / "mistralai/Mistral-Nemo-Base-2407", "Mistral-Nemo-Base-2407"),
    (AXON_DIR / "mistral/Ministral-8B-Instruct-2410.axon", MODELS_DIR / "mistralai/Ministral-8B-Instruct-2410", "Ministral-8B-Instruct-2410"),
    (AXON_DIR / "llama3/Llama-3.1-8B.axon", MODELS_DIR / "meta-llama/Llama-3.1-8B", "Llama-3.1-8B"),
    (AXON_DIR / "gemma2/gemma-2-9b.axon", MODELS_DIR / "google/gemma-2-9b", "gemma-2-9b"),
    (AXON_DIR / "gemma3/gemma-3-12b.axon", MODELS_DIR / "google/gemma-3-12b-pt", "gemma-3-12b"),
    (AXON_DIR / "starcoder2/starcoder2-7b.axon", MODELS_DIR / "bigcode/starcoder2-7b", "starcoder2-7b"),
    (AXON_DIR / "phi3minimedium/Phi-3-mini-128k-instruct.axon", MODELS_DIR / "microsoft/Phi-3-mini-128k-instruct", "Phi-3-mini-128k"),
    (AXON_DIR / "phi3minimedium/Phi-3-medium-4k-instruct.axon", MODELS_DIR / "microsoft/Phi-3-medium-4k-instruct", "Phi-3-medium-4k"),
    (AXON_DIR / "phi3minimedium/Phi-3-medium-128k-instruct.axon", MODELS_DIR / "microsoft/Phi-3-medium-128k-instruct", "Phi-3-medium-128k"),
    (AXON_DIR / "phi4/Phi-4-mini-reasoning.axon", MODELS_DIR / "microsoft/Phi-4-mini-reasoning", "Phi-4-mini-reasoning"),
    (AXON_DIR / "olmo2/OLMo-2-1124-13B.axon", MODELS_DIR / "allenai/OLMo-2-1124-13B", "OLMo-2-1124-13B"),
    (AXON_DIR / "pleias/Pleias-3b-Preview.axon", MODELS_DIR / "PleIAs/Pleias-3b-Preview", "Pleias-3b-Preview"),
    (AXON_DIR / "pleias/Pleias-RAG-1B.axon", MODELS_DIR / "PleIAs/Pleias-RAG-1B", "Pleias-RAG-1B"),
    (AXON_DIR / "pleias/Pleias-RAG-350M.axon", MODELS_DIR / "PleIAs/Pleias-RAG-350M", "Pleias-RAG-350M"),
    (AXON_DIR / "glm/glm-edge-4b-chat.axon", MODELS_DIR / "zai-org/glm-edge-4b-chat", "glm-edge-4b-chat"),
    (AXON_DIR / "glm/GLM-4-9B-0414.axon", MODELS_DIR / "zai-org/GLM-4-9B-0414", "GLM-4-9B-0414"),
    # --- Expanded: new architecture families ---
    (AXON_DIR / "opt/opt-1.3b.axon", MODELS_DIR / "facebook/opt-1.3b", "opt-1.3b"),
    (AXON_DIR / "opt/opt-6.7b.axon", MODELS_DIR / "facebook/opt-6.7b", "opt-6.7b"),
    (AXON_DIR / "opt/opt-13b.axon", MODELS_DIR / "facebook/opt-13b", "opt-13b"),
    (AXON_DIR / "xglm/xglm-564M.axon", MODELS_DIR / "facebook/xglm-564M", "xglm-564M"),
    (AXON_DIR / "xglm/xglm-1.7B.axon", MODELS_DIR / "facebook/xglm-1.7B", "xglm-1.7B"),
    (AXON_DIR / "xglm/xglm-2.9B.axon", MODELS_DIR / "facebook/xglm-2.9B", "xglm-2.9B"),
    (AXON_DIR / "gpt-j/gpt-j-6b.axon", MODELS_DIR / "EleutherAI/gpt-j-6b", "gpt-j-6b"),
    (AXON_DIR / "gpt-neox/pythia-6.9b.axon", MODELS_DIR / "EleutherAI/pythia-6.9b", "pythia-6.9b"),
    (AXON_DIR / "gpt-neox/gpt-neox-20b.axon", MODELS_DIR / "EleutherAI/gpt-neox-20b", "gpt-neox-20b"),
    (AXON_DIR / "apertus/Apertus-8B-2509.axon", MODELS_DIR / "swiss-ai/Apertus-8B-2509", "Apertus-8B-2509"),
    (AXON_DIR / "olmo3/Olmo-3-1025-7B.axon", MODELS_DIR / "allenai/Olmo-3-1025-7B", "Olmo-3-1025-7B"),
    (AXON_DIR / "qwen3/Qwen3-14B.axon", MODELS_DIR / "Qwen/Qwen3-14B", "Qwen3-14B"),
    (AXON_DIR / "hrm_text/HRM-Text-1B.axon", MODELS_DIR / "sapientinc/HRM-Text-1B", "HRM-Text-1B"),
    (AXON_DIR / "gpt-oss/gpt-oss-20b.axon", MODELS_DIR / "openai/gpt-oss-20b", "gpt-oss-20b"),
    (AXON_DIR / "phi3small/Phi-3-small-8k-instruct.axon", MODELS_DIR / "microsoft/Phi-3-small-8k-instruct", "Phi-3-small-8k"),
    (AXON_DIR / "nanochat/generic-nanochat.axon", MODELS_DIR / "nanochat-students/nanochat-d20", "nanochat-d20"),
    (AXON_DIR / "helium/generic-helium.axon", MODELS_DIR / "kyutai/helium-1-preview-2b", "helium-1-preview-2b"),
    # --- Expanded: MoE models ---
    (AXON_DIR / "granitemoe/PowerMoE-3b.axon", MODELS_DIR / "ibm-research/PowerMoE-3b", "PowerMoE-3b"),
    (AXON_DIR / "olmoe/OLMoE-1B-7B-0924.axon", MODELS_DIR / "allenai/OLMoE-1B-7B-0924", "OLMoE-1B-7B"),
    (AXON_DIR / "deepseek/deepseek-moe-16b-base.axon", MODELS_DIR / "deepseek-ai/deepseek-moe-16b-base", "deepseek-moe-16b"),
    (AXON_DIR / "qwen3-moe/Qwen3-30B-A3B.axon", MODELS_DIR / "Qwen/Qwen3-30B-A3B", "Qwen3-30B-A3B"),
    # --- Expanded: other architectures ---
    (AXON_DIR / "comma/comma-v0.1-1t.axon", MODELS_DIR / "common-pile/comma-v0.1-1t", "comma-v0.1-1t"),
    (AXON_DIR / "flexolmo/Flex-code-2x7B-1T.axon", MODELS_DIR / "allenai/Flex-code-2x7B-1T", "Flex-code-2x7B-1T"),
    (AXON_DIR / "mistral3/Ministral-3-8B-Base-2512.axon", MODELS_DIR / "mistralai/Ministral-3-8B-Base-2512", "Ministral-3-8B"),
    (AXON_DIR / "mistral3/Ministral-3-14B-Base-2512.axon", MODELS_DIR / "mistralai/Ministral-3-14B-Base-2512", "Ministral-3-14B"),
    (AXON_DIR / "gemma3/gemma-3-27b.axon", MODELS_DIR / "google/gemma-3-27b-pt", "gemma-3-27b"),
    (AXON_DIR / "starcoder2/starcoder2-15b.axon", MODELS_DIR / "bigcode/starcoder2-15b", "starcoder2-15b"),
]

results = []
for axon_file, model_dir, name in TESTS:
    if not axon_file.exists():
        print(f"\nSKIP {name}: axon file not found: {axon_file}")
        results.append((name, "SKIP", "axon file not found", None, None))
        continue
    if not model_dir.exists():
        print(f"\nSKIP {name}: model dir not found: {model_dir}")
        results.append((name, "SKIP", "model dir not found", None, None))
        continue

    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"  axon: {axon_file}")
    print(f"  model: {model_dir}")
    print(f"{'='*60}")

    try:
        result = run_axon_test(
            axon_file=axon_file,
            weights=model_dir,
            hf_model_dir=model_dir,
            device="cuda",
            dtype="bfloat16",
            axon_backend="codegen2-vllm",
            model_task="auto",
            benchmark_mode="forward",
            max_len=128,
            text=["The future of AI is"],
            compile_axon=False,
            compile_hf=False,
            optimize_graph=True,
            forward_warmup=1,
            forward_repeat=3,
            generate_warmup=1,
            generate_repeat=3,
        )
        top1 = result.get("top1_eq", False)
        max_diff = result.get("max_diff", float('inf'))
        hf_time = result.get("hf_time", 0)
        axon_time = result.get("axon_time", 0)
        speed_ratio = axon_time / hf_time if hf_time > 0 else float('inf')
        gen = result.get("gen")
        gen_match = ""
        if gen is not None:
            gen_match = f" gen_shape={list(gen.shape) if hasattr(gen, 'shape') else 'N/A'}"
        status = "PASS" if top1 else "FAIL"
        print(f"  RESULT: {status} top1={top1} max_diff={max_diff:.4f} "
              f"hf_time={hf_time:.4f}s axon_time={axon_time:.4f}s speed_ratio={speed_ratio:.3f}{gen_match}")
        results.append((name, status, f"top1={top1} max_diff={max_diff:.4f} speed={speed_ratio:.3f}", hf_time, axon_time))
    except Exception as e:
        err_msg = f"{type(e).__name__}: {str(e)[:200]}"
        print(f"  ERROR: {err_msg}")
        traceback.print_exc()
        results.append((name, "ERROR", err_msg, None, None))

print(f"\n\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"{'Model':<25} {'Status':<8} {'Details':<50}")
print(f"{'-'*25} {'-'*8} {'-'*50}")
for name, status, details, hf_t, axon_t in results:
    print(f"{name:<25} {status:<8} {details:<50}")
