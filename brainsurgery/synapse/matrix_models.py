from __future__ import annotations

import concurrent.futures
import fcntl
import json
import os
import random
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

_HF_API = "https://huggingface.co/api/models"
_HF_RESOLVE = "https://huggingface.co/{repo_id}/resolve/main/{filename}"
_ESSENTIAL_TEXT_FILES = {
    "config.json",
    "generation_config.json",
    "generator_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "spm.model",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
}
_DEFAULT_PARALLEL_WORKERS = 4
_DEFAULT_MAX_RETRIES = 20
_DEFAULT_BACKOFF_INITIAL_S = 2.0
_DEFAULT_BACKOFF_MAX_S = 60.0
_HF_SIBLINGS_CACHE: dict[str, list["HFSibling"]] = {}
_HF_CONTENT_LENGTH_CACHE: dict[tuple[str, str], int | None] = {}


@dataclass(frozen=True)
class ModelDownloadSpec:
    local_dir: str
    repo_id: str
    require_tokenizer: bool = True


@dataclass(frozen=True)
class HFSibling:
    rfilename: str
    size_bytes: int | None = None


MODEL_SPECS: dict[str, ModelDownloadSpec] = {
    "albert": ModelDownloadSpec(local_dir="albert", repo_id="albert/albert-base-v2"),
    "apertus_8b": ModelDownloadSpec(local_dir="apertus_8b", repo_id="swiss-ai/Apertus-8B-2509"),
    "bart_base": ModelDownloadSpec(local_dir="bart_base", repo_id="facebook/bart-base"),
    "bert": ModelDownloadSpec(local_dir="bert", repo_id="google-bert/bert-base-uncased"),
    "black_mamba": ModelDownloadSpec(
        local_dir="black_mamba",
        repo_id="Zyphra/BlackMamba-2.8B",
        require_tokenizer=False,
    ),
    "black_mamba_2_8b": ModelDownloadSpec(
        local_dir="black_mamba_2_8b",
        repo_id="Zyphra/BlackMamba-2.8B",
        require_tokenizer=False,
    ),
    "camembert": ModelDownloadSpec(local_dir="camembert", repo_id="camembert-base"),
    "comma": ModelDownloadSpec(local_dir="comma", repo_id="common-pile/comma-v0.1-1t"),
    "deberta_v2": ModelDownloadSpec(
        local_dir="deberta_v2",
        repo_id="microsoft/deberta-v3-xsmall",
    ),
    "deepseek_v2_lite": ModelDownloadSpec(
        local_dir="deepseek_v2_lite",
        repo_id="deepseek-ai/DeepSeek-V2-Lite",
    ),
    "dfm_decoder": ModelDownloadSpec(
        local_dir="dfm_decoder",
        repo_id="danish-foundation-models/dfm-decoder-open-v0-7b-pt",
    ),
    "distilbert": ModelDownloadSpec(
        local_dir="distilbert",
        repo_id="distilbert/distilbert-base-uncased",
    ),
    "electra": ModelDownloadSpec(
        local_dir="electra",
        repo_id="google/electra-base-generator",
    ),
    "falcon_rw_1b": ModelDownloadSpec(local_dir="falcon_rw_1b", repo_id="tiiuae/falcon-rw-1b"),
    "flexmath": ModelDownloadSpec(local_dir="flexmath", repo_id="allenai/Flex-math-2x7B-1T"),
    "flexolmo": ModelDownloadSpec(local_dir="flexolmo", repo_id="allenai/FlexOlmo-7B"),
    "flexolmo_7x7b_1t": ModelDownloadSpec(
        local_dir="flexolmo_7x7b_1t", repo_id="allenai/FlexOlmo-7x7B-1T"
    ),
    "gemma3": ModelDownloadSpec(local_dir="gemma3", repo_id="google/gemma-3-270m"),
    "gemma3_1b": ModelDownloadSpec(local_dir="gemma3_1b", repo_id="google/gemma-3-1b-pt"),
    "gemma3_4b": ModelDownloadSpec(local_dir="gemma3_4b", repo_id="google/gemma-3-4b-pt"),
    "gemma3_12b": ModelDownloadSpec(local_dir="gemma3_12b", repo_id="google/gemma-3-12b-pt"),
    "gemma3_27b": ModelDownloadSpec(local_dir="gemma3_27b", repo_id="google/gemma-3-27b-pt"),
    "gemma4_26b_a4b": ModelDownloadSpec(
        local_dir="gemma4_26b_a4b",
        repo_id="google/gemma-4-26B-A4B",
    ),
    "gemma4_31b": ModelDownloadSpec(local_dir="gemma4_31b", repo_id="google/gemma-4-31B"),
    "gemma4_e2b": ModelDownloadSpec(local_dir="gemma4_e2b", repo_id="google/gemma-4-E2B"),
    "gemma4_e4b": ModelDownloadSpec(local_dir="gemma4_e4b", repo_id="google/gemma-4-E4B"),
    "glm_4_5_air": ModelDownloadSpec(local_dir="glm_4_5_air", repo_id="zai-org/GLM-4.5-Air"),
    "gpt2": ModelDownloadSpec(local_dir="gpt2", repo_id="openai-community/gpt2"),
    "gpt_oss_20b": ModelDownloadSpec(local_dir="gpt_oss_20b", repo_id="openai/gpt-oss-20b"),
    "gpt_oss_120b": ModelDownloadSpec(
        local_dir="gpt_oss_120b",
        repo_id="openai/gpt-oss-120b",
    ),
    "jamba_3b": ModelDownloadSpec(
        local_dir="jamba_3b", repo_id="ai21labs/AI21-Jamba-Reasoning-3B"
    ),
    "llama3_2_1b": ModelDownloadSpec(local_dir="llama3_2_1b", repo_id="meta-llama/Llama-3.2-1B"),
    "longformer": ModelDownloadSpec(
        local_dir="longformer",
        repo_id="allenai/longformer-base-4096",
    ),
    "mamba_2_8b_hf": ModelDownloadSpec(
        local_dir="mamba_2_8b_hf", repo_id="state-spaces/mamba-2.8b-hf"
    ),
    "marian_en_de": ModelDownloadSpec(
        local_dir="marian_en_de", repo_id="Helsinki-NLP/opus-mt-en-de"
    ),
    "mbart_large_50_m2m": ModelDownloadSpec(
        local_dir="mbart_large_50_m2m",
        repo_id="facebook/mbart-large-50-many-to-many-mmt",
    ),
    "mistral_7b_v0_1": ModelDownloadSpec(
        local_dir="mistral_7b_v0_1", repo_id="mistralai/Mistral-7B-v0.1"
    ),
    "modernbert": ModelDownloadSpec(local_dir="modernbert", repo_id="answerdotai/ModernBERT-base"),
    "mt5_base": ModelDownloadSpec(local_dir="mt5_base", repo_id="google/mt5-base"),
    "mt5_large": ModelDownloadSpec(local_dir="mt5_large", repo_id="google/mt5-large"),
    "mt5_small": ModelDownloadSpec(local_dir="mt5_small", repo_id="google/mt5-small"),
    "mt5_xl": ModelDownloadSpec(local_dir="mt5_xl", repo_id="google/mt5-xl"),
    "mt5_xxl": ModelDownloadSpec(local_dir="mt5_xxl", repo_id="google/mt5-xxl"),
    "nemotron3": ModelDownloadSpec(
        local_dir="nemotron3",
        repo_id="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    ),
    "olmo3_1025_7b": ModelDownloadSpec(
        local_dir="olmo3_1025_7b",
        repo_id="allenai/Olmo-3-1025-7B",
    ),
    "olmo3_7b_instruct": ModelDownloadSpec(
        local_dir="olmo3_7b_instruct",
        repo_id="allenai/Olmo-3-7B-Instruct",
    ),
    "olmo3_7b_think": ModelDownloadSpec(
        local_dir="olmo3_7b_think",
        repo_id="allenai/Olmo-3-7B-Think",
    ),
    "olmo_2_1b": ModelDownloadSpec(local_dir="olmo_2_1b", repo_id="allenai/OLMo-2-0425-1B"),
    "olmo_2_7b": ModelDownloadSpec(local_dir="olmo_2_7b", repo_id="allenai/OLMo-2-1124-7B"),
    "olmo_2_13b": ModelDownloadSpec(
        local_dir="olmo_2_13b", repo_id="allenai/OLMo-2-1124-13B"
    ),
    "olmoe_1b_7b_0924": ModelDownloadSpec(
        local_dir="olmoe_1b_7b_0924", repo_id="allenai/OLMoE-1B-7B-0924"
    ),
    "phi3_medium_128k_instruct": ModelDownloadSpec(
        local_dir="phi3_medium_128k_instruct",
        repo_id="microsoft/Phi-3-medium-128k-instruct",
    ),
    "phi3_medium_4k_instruct": ModelDownloadSpec(
        local_dir="phi3_medium_4k_instruct",
        repo_id="microsoft/Phi-3-medium-4k-instruct",
    ),
    "phi3_mini_128k_instruct": ModelDownloadSpec(
        local_dir="phi3_mini_128k_instruct",
        repo_id="microsoft/Phi-3-mini-128k-instruct",
    ),
    "phi3_mini_4k_instruct": ModelDownloadSpec(
        local_dir="phi3_mini_4k_instruct",
        repo_id="microsoft/Phi-3-mini-4k-instruct",
    ),
    "phi3_small_128k_instruct": ModelDownloadSpec(
        local_dir="phi3_small_128k_instruct",
        repo_id="microsoft/Phi-3-small-128k-instruct",
    ),
    "phi3_small_8k_instruct": ModelDownloadSpec(
        local_dir="phi3_small_8k_instruct",
        repo_id="microsoft/Phi-3-small-8k-instruct",
    ),
    "qwen2_5_0_5b": ModelDownloadSpec(local_dir="qwen2_5_0_5b", repo_id="Qwen/Qwen2.5-0.5B"),
    "roberta": ModelDownloadSpec(local_dir="roberta", repo_id="FacebookAI/roberta-base"),
    "smollm_135m": ModelDownloadSpec(
        local_dir="smollm_135m",
        repo_id="HuggingFaceTB/SmolLM-135M",
    ),
    "smollm_1_7b": ModelDownloadSpec(
        local_dir="smollm_1_7b",
        repo_id="HuggingFaceTB/SmolLM-1.7B",
    ),
    "smollm_360m": ModelDownloadSpec(
        local_dir="smollm_360m",
        repo_id="HuggingFaceTB/SmolLM-360M",
    ),
    "smollm2_135m": ModelDownloadSpec(
        local_dir="smollm2_135m",
        repo_id="HuggingFaceTB/SmolLM2-135M",
    ),
    "smollm2_1_7b": ModelDownloadSpec(
        local_dir="smollm2_1_7b",
        repo_id="HuggingFaceTB/SmolLM2-1.7B",
    ),
    "smollm2_360m": ModelDownloadSpec(
        local_dir="smollm2_360m",
        repo_id="HuggingFaceTB/SmolLM2-360M",
    ),
    "smollm3_3b": ModelDownloadSpec(
        local_dir="smollm3_3b",
        repo_id="HuggingFaceTB/SmolLM3-3B",
    ),
    "smollm3_3b_base": ModelDownloadSpec(
        local_dir="smollm3_3b_base",
        repo_id="HuggingFaceTB/SmolLM3-3B-Base",
    ),
    "t5_11b": ModelDownloadSpec(local_dir="t5_11b", repo_id="google-t5/t5-11b"),
    "t5_3b": ModelDownloadSpec(local_dir="t5_3b", repo_id="google-t5/t5-3b"),
    "t5_base": ModelDownloadSpec(local_dir="t5_base", repo_id="google-t5/t5-base"),
    "t5_large": ModelDownloadSpec(local_dir="t5_large", repo_id="google-t5/t5-large"),
    "t5_small": ModelDownloadSpec(local_dir="t5_small", repo_id="google-t5/t5-small"),
    "t5gemma2_270m": ModelDownloadSpec(
        local_dir="t5gemma2_270m", repo_id="google/t5gemma-2-270m-270m"
    ),
    "t5gemma_s_s_ul2": ModelDownloadSpec(
        local_dir="t5gemma_s_s_ul2", repo_id="google/t5gemma-s-s-ul2"
    ),
    "xlm_roberta": ModelDownloadSpec(
        local_dir="xlm_roberta",
        repo_id="FacebookAI/xlm-roberta-base",
    ),
}

MATRIX_AXON_MODEL_DIRS: dict[str, tuple[str, ...]] = {
    "albert": ("albert",),
    "apertus_8b": ("apertus_8b",),
    "bart": ("bart_base",),
    "bert": ("bert",),
    "black_mamba": ("black_mamba", "black_mamba_2_8b"),
    "deberta_v2": ("deberta_v2",),
    "deepseek_v2_lite": ("deepseek_v2_lite",),
    "dfm_decoder": ("dfm_decoder", "comma"),
    "distilbert": ("distilbert",),
    "electra": ("electra",),
    "falcon_rw_1b": ("falcon_rw_1b",),
    "flexolmo": ("flexmath", "flexolmo", "flexolmo_7x7b_1t"),
    "gemma3": ("gemma3", "gemma3_4b", "gemma3_12b", "gemma3_27b"),
    "gemma3_270m": ("gemma3",),
    "gemma4_dense": ("gemma4_31b",),
    "gemma4_e": ("gemma4_e2b", "gemma4_e4b"),
    "gemma4_moe": ("gemma4_26b_a4b",),
    "gemma_1b": ("gemma3_1b",),
    "glm_4_5_air": ("glm_4_5_air",),
    "gpt2": ("gpt2",),
    "gpt2_kv": ("gpt2",),
    "gpt_oss_20b": ("gpt_oss_20b", "gpt_oss_120b"),
    "jamba_3b": ("jamba_3b",),
    "llama3_2_1b": ("llama3_2_1b",),
    "longformer": ("longformer",),
    "mamba_2_8b": ("mamba_2_8b_hf",),
    "marian": ("marian_en_de",),
    "mbart": ("mbart_large_50_m2m",),
    "mistral_7b_v0_1": ("mistral_7b_v0_1",),
    "modernbert": ("modernbert",),
    "mt5": ("mt5_small", "mt5_base", "mt5_large", "mt5_xl", "mt5_xxl"),
    "nemotron3": ("nemotron3",),
    "olmo2": ("olmo_2_1b", "olmo_2_7b", "olmo_2_13b"),
    "olmo3": ("olmo3_1025_7b", "olmo3_7b_instruct", "olmo3_7b_think"),
    "olmo_2_1b": ("olmo_2_1b",),
    "olmoe_1b_7b_0924": ("olmoe_1b_7b_0924",),
    "phi3_mini_4k_instruct": ("phi3_mini_4k_instruct",),
    "phi3minimedium": (
        "phi3_mini_4k_instruct",
        "phi3_mini_128k_instruct",
        "phi3_medium_4k_instruct",
        "phi3_medium_128k_instruct",
    ),
    "phi3small": ("phi3_small_8k_instruct", "phi3_small_128k_instruct"),
    "qwen2_5_0_5b": ("qwen2_5_0_5b",),
    "roberta": ("roberta", "camembert", "xlm_roberta"),
    "smollm": (
        "smollm_135m",
        "smollm_360m",
        "smollm_1_7b",
        "smollm2_135m",
        "smollm2_360m",
        "smollm2_1_7b",
    ),
    "smollm3": ("smollm3_3b", "smollm3_3b_base"),
    "smollm_135m": ("smollm_135m",),
    "t5": ("t5_small", "t5_base", "t5_large", "t5_3b", "t5_11b"),
    "t5_small": ("t5_small",),
    "t5gemma": ("t5gemma_s_s_ul2",),
    "t5gemma2": ("t5gemma2_270m",),
}

MATRIX_AXON_MODEL_DIR_PAIRS: list[tuple[str, str]] = [
    (axon_stem, model_dir)
    for axon_stem, model_dirs in MATRIX_AXON_MODEL_DIRS.items()
    for model_dir in model_dirs
]


def all_matrix_model_dirs() -> tuple[str, ...]:
    return tuple(sorted({model_dir for _, model_dir in MATRIX_AXON_MODEL_DIR_PAIRS}))


def _status(status_cb: Callable[[str], None] | None, message: str) -> None:
    if status_cb is not None:
        status_cb(message)


def _run_curl(
    *,
    url: str,
    out_path: Path,
    headers: list[str] | None = None,
    resume: bool,
    cwd: Path,
) -> None:
    cmd = ["curl", "-fL"]
    if resume:
        cmd.extend(["-C", "-"])
    if headers:
        for header in headers:
            cmd.extend(["-H", header])
    cmd.extend(["-o", str(out_path), url])
    run = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    if run.returncode != 0:
        raise RuntimeError(f"curl failed for {url}\nstdout:\n{run.stdout}\nstderr:\n{run.stderr}")


def _head_content_length(*, url: str, headers: list[str] | None = None, cwd: Path) -> int | None:
    cmd = ["curl", "-fsIL"]
    if headers:
        for header in headers:
            cmd.extend(["-H", header])
    cmd.append(url)
    run = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    if run.returncode != 0:
        return None
    content_length: int | None = None
    for raw_line in run.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.lower().startswith("content-length:"):
            value = line.split(":", 1)[1].strip()
            try:
                parsed = int(value)
            except ValueError:
                continue
            if parsed >= 0:
                content_length = parsed
    return content_length


def _download_with_retry(
    *,
    url: str,
    out_path: Path,
    headers: list[str],
    cwd: Path,
    status_cb: Callable[[str], None] | None,
    model_name: str,
    filename: str,
    max_retries: int,
    backoff_initial_s: float,
    backoff_max_s: float,
) -> None:
    attempt = 0
    while True:
        attempt += 1
        try:
            _run_curl(
                url=url,
                out_path=out_path,
                headers=headers,
                resume=True,
                cwd=cwd,
            )
            return
        except RuntimeError as exc:
            if attempt >= max_retries:
                raise RuntimeError(
                    f"{model_name}: failed downloading {filename} after {attempt} attempts"
                ) from exc
            sleep_s = min(backoff_max_s, backoff_initial_s * (2 ** (attempt - 1)))
            sleep_s += random.uniform(0.0, 0.5)
            _status(
                status_cb,
                (
                    f"{model_name}: retry {attempt}/{max_retries} for {filename} "
                    f"after error; sleeping {sleep_s:.1f}s"
                ),
            )
            time.sleep(sleep_s)


def _parallel_worker_count(num_items: int) -> int:
    env_value = os.environ.get("MODEL_DOWNLOAD_WORKERS")
    workers = _DEFAULT_PARALLEL_WORKERS
    if env_value:
        try:
            workers = int(env_value)
        except ValueError:
            workers = _DEFAULT_PARALLEL_WORKERS
    workers = max(1, workers)
    return min(workers, max(1, num_items))


def _is_valid_safetensors_file(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        from safetensors import safe_open
    except Exception:
        return True
    try:
        with safe_open(str(path), framework="pt") as handle:
            _ = list(handle.keys())
        return True
    except Exception:
        return False


def _auth_headers() -> list[str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        return []
    return [f"Authorization: Bearer {token}"]


def _load_hf_sibling_entries(*, repo_id: str, cwd: Path) -> list[HFSibling]:
    cached = _HF_SIBLINGS_CACHE.get(repo_id)
    if cached is not None:
        return cached
    headers = _auth_headers()
    api_url = f"{_HF_API}/{repo_id}"
    tmp = cwd / ".tmp_hf_model_api.json"
    _run_curl(url=api_url, out_path=tmp, headers=headers, resume=False, cwd=cwd)
    try:
        payload = json.loads(tmp.read_text(encoding="utf-8"))
    finally:
        tmp.unlink(missing_ok=True)
    siblings = payload.get("siblings")
    if not isinstance(siblings, list):
        raise RuntimeError(f"Unexpected HF API payload for {repo_id}: missing siblings")
    out: list[HFSibling] = []
    for item in siblings:
        if not isinstance(item, dict):
            continue
        name = item.get("rfilename")
        if not isinstance(name, str):
            continue
        size_value = item.get("size")
        size_bytes = size_value if isinstance(size_value, int) and size_value >= 0 else None
        out.append(HFSibling(rfilename=name, size_bytes=size_bytes))
    _HF_SIBLINGS_CACHE[repo_id] = out
    return out


def _load_hf_siblings(*, repo_id: str, cwd: Path) -> list[str]:
    return [entry.rfilename for entry in _load_hf_sibling_entries(repo_id=repo_id, cwd=cwd)]


def estimate_remote_param_count_lower_bound(
    *,
    repo_root: Path,
    spec: ModelDownloadSpec,
) -> int | None:
    sibling_entries = _load_hf_sibling_entries(repo_id=spec.repo_id, cwd=repo_root)
    shard_entries = [
        entry
        for entry in sibling_entries
        if "/" not in entry.rfilename and entry.rfilename.endswith(".safetensors")
    ]
    if not shard_entries:
        shard_entries = [
            entry for entry in sibling_entries if "/" not in entry.rfilename and entry.rfilename.endswith(".bin")
        ]
        if not shard_entries:
            return None
    headers = _auth_headers()
    total_bytes = 0
    for entry in shard_entries:
        size_bytes = entry.size_bytes
        if size_bytes is None:
            cache_key = (spec.repo_id, entry.rfilename)
            if cache_key not in _HF_CONTENT_LENGTH_CACHE:
                _HF_CONTENT_LENGTH_CACHE[cache_key] = _head_content_length(
                    url=_HF_RESOLVE.format(repo_id=spec.repo_id, filename=entry.rfilename),
                    headers=headers,
                    cwd=repo_root,
                )
            size_bytes = _HF_CONTENT_LENGTH_CACHE[cache_key]
        if size_bytes is None:
            return None
        total_bytes += int(size_bytes)
    # This is a safe lower bound assuming the largest common dtype footprint (float32).
    return (total_bytes + 3) // 4


def _is_complete_model_dir(model_dir: Path, *, require_tokenizer: bool) -> bool:
    index_path = model_dir / "model.safetensors.index.json"
    single_path = model_dir / "model.safetensors"
    pytorch_bin_path = model_dir / "pytorch_model.bin"
    pt_files = (
        sorted(model_dir.glob("*.pt"))
        + sorted(model_dir.glob("*.pth"))
        + sorted(model_dir.glob("*.bin"))
    )
    safetensor_files = sorted(model_dir.glob("*.safetensors"))

    if pt_files and safetensor_files:
        return False

    has_weights = False
    if index_path.exists():
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = payload.get("weight_map")
            if not isinstance(weight_map, dict):
                return False
            shard_names = {str(v) for v in weight_map.values()}
            if not shard_names:
                return False
            has_weights = all((model_dir / shard).exists() for shard in shard_names)
        except json.JSONDecodeError:
            return False
    elif single_path.exists():
        has_weights = True
    elif pytorch_bin_path.exists():
        has_weights = True

    if not has_weights:
        return False

    if not (model_dir / "config.json").exists():
        return False
    if require_tokenizer:
        has_tokenizer = (
            (model_dir / "tokenizer_config.json").exists()
            or (model_dir / "tokenizer.json").exists()
            or (model_dir / "tokenizer.model").exists()
            or (model_dir / "spm.model").exists()
            or ((model_dir / "vocab.json").exists() and (model_dir / "merges.txt").exists())
            or any(model_dir.glob("*.tiktoken"))
            or any(model_dir.glob("tokenization*.py"))
        )
        if not has_tokenizer:
            return False

    return True


def _normalize_config_rope_numeric_fields(model_dir: Path) -> None:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(payload, dict):
        return

    changed = False

    def _normalize(mapping: object) -> None:
        nonlocal changed
        if not isinstance(mapping, dict):
            return
        for key in ("factor", "beta_fast", "beta_slow", "mscale", "mscale_all_dim"):
            value = mapping.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                mapping[key] = float(value)
                changed = True

    _normalize(payload.get("rope_scaling"))
    _normalize(payload.get("rope_parameters"))
    original_ctx = payload.get("original_max_position_embeddings")
    if isinstance(original_ctx, int) and not isinstance(original_ctx, bool):
        for field_name in ("rope_scaling", "rope_parameters"):
            field = payload.get(field_name)
            if not isinstance(field, dict):
                continue
            rope_type = field.get("rope_type", field.get("type"))
            if rope_type not in {"longrope", "su"}:
                continue
            if "original_max_position_embeddings" not in field:
                field["original_max_position_embeddings"] = original_ctx
                changed = True

    if changed:
        config_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


def _normalize_local_weight_format(model_dir: Path) -> None:
    safetensor_files = sorted(model_dir.glob("*.safetensors"))
    pt_files = (
        sorted(model_dir.glob("*.pt"))
        + sorted(model_dir.glob("*.pth"))
        + sorted(model_dir.glob("*.bin"))
    )
    if safetensor_files and pt_files:
        for stale in pt_files:
            stale.unlink(missing_ok=True)
        return
    if safetensor_files or not pt_files:
        return

    try:
        from safetensors.torch import save_file as save_safetensors_file
    except Exception as exc:
        raise RuntimeError("safetensors is required to normalize PyTorch checkpoints") from exc

    def _extract_tensor_mapping(payload: object) -> dict[str, torch.Tensor] | None:
        if not isinstance(payload, dict):
            return None
        if all(isinstance(k, str) and torch.is_tensor(v) for k, v in payload.items()):
            return {str(k): v for k, v in payload.items()}
        state_dict = payload.get("state_dict")
        if isinstance(state_dict, dict) and all(
            isinstance(k, str) and torch.is_tensor(v) for k, v in state_dict.items()
        ):
            return {str(k): v for k, v in state_dict.items()}
        if len(payload) == 1:
            only_value = next(iter(payload.values()))
            if isinstance(only_value, dict) and all(
                isinstance(k, str) and torch.is_tensor(v) for k, v in only_value.items()
            ):
                return {str(k): v for k, v in only_value.items()}
        return None

    for pt_path in pt_files:
        payload = torch.load(str(pt_path), map_location="cpu", weights_only=False)
        tensor_payload = _extract_tensor_mapping(payload)
        if tensor_payload is None:
            raise RuntimeError(f"Unsupported PyTorch checkpoint payload in {pt_path}")

        tensor_map: dict[str, torch.Tensor] = {}
        for key, value in tensor_payload.items():
            tensor_map[key] = value.detach().cpu().clone().contiguous()
        if not tensor_map:
            raise RuntimeError(f"No tensors found in PyTorch checkpoint {pt_path}")

        target_name = (
            "model.safetensors" if pt_path.name == "pytorch_model.bin" else f"{pt_path.stem}.safetensors"
        )
        save_safetensors_file(tensor_map, str(model_dir / target_name))
        pt_path.unlink(missing_ok=True)


def ensure_model_downloaded(
    *,
    repo_root: Path,
    spec: ModelDownloadSpec,
    status_cb: Callable[[str], None] | None = None,
) -> Path:
    if shutil.which("curl") is None:
        raise RuntimeError("curl is required to download test models")

    model_dir = repo_root / "models" / spec.local_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    lock_path = model_dir / ".download.lock"
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            _normalize_local_weight_format(model_dir)

            if _is_complete_model_dir(model_dir, require_tokenizer=spec.require_tokenizer):
                _normalize_config_rope_numeric_fields(model_dir)
                _status(status_cb, f"{spec.local_dir}: already complete, skipping download")
                return model_dir

            _status(status_cb, f"{spec.local_dir}: fetching sibling manifest from {spec.repo_id}")
            siblings = _load_hf_siblings(repo_id=spec.repo_id, cwd=repo_root)

            shard_files = sorted(
                name for name in siblings if "/" not in name and name.endswith(".safetensors")
            )
            pytorch_bin_files = sorted(
                name for name in siblings if "/" not in name and name.endswith(".bin")
            )
            index_files = [name for name in siblings if name == "model.safetensors.index.json"]
            if shard_files:
                selected_weight_files = [*index_files, *shard_files]
                for stale in [*model_dir.glob("*.pt"), *model_dir.glob("*.pth"), *model_dir.glob("*.bin")]:
                    stale.unlink(missing_ok=True)
            else:
                selected_weight_files = pytorch_bin_files
                for stale in [*model_dir.glob("*.safetensors"), model_dir / "model.safetensors.index.json"]:
                    stale.unlink(missing_ok=True)
            selected_files = set(selected_weight_files)
            selected_files.update(
                name
                for name in siblings
                if (
                    Path(name).name in _ESSENTIAL_TEXT_FILES
                    or Path(name).name.endswith(".tiktoken")
                    or (
                        Path(name).name.endswith(".py")
                        and Path(name).name.startswith(("configuration_", "modeling_", "tokenization_"))
                    )
                )
            )

            headers = _auth_headers()
            pending_files: list[str] = []
            for name in sorted(selected_files):
                target = model_dir / name
                if name.endswith(".safetensors"):
                    if not _is_valid_safetensors_file(target):
                        pending_files.append(name)
                    continue
                if not target.exists():
                    pending_files.append(name)
            if not pending_files:
                if not _is_complete_model_dir(model_dir, require_tokenizer=spec.require_tokenizer):
                    raise RuntimeError(
                        f"Model download incomplete for {spec.local_dir} ({spec.repo_id}) at {model_dir}"
                    )
                _normalize_config_rope_numeric_fields(model_dir)
                _status(status_cb, f"{spec.local_dir}: download complete")
                return model_dir

            text_files = [name for name in pending_files if name in _ESSENTIAL_TEXT_FILES]
            weight_files = [name for name in pending_files if name not in _ESSENTIAL_TEXT_FILES]

            max_retries = int(os.environ.get("MODEL_DOWNLOAD_MAX_RETRIES", _DEFAULT_MAX_RETRIES))
            backoff_initial_s = float(
                os.environ.get("MODEL_DOWNLOAD_BACKOFF_INITIAL_S", _DEFAULT_BACKOFF_INITIAL_S)
            )
            backoff_max_s = float(
                os.environ.get("MODEL_DOWNLOAD_BACKOFF_MAX_S", _DEFAULT_BACKOFF_MAX_S)
            )

            for name in text_files:
                dst = model_dir / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                _status(status_cb, f"{spec.local_dir}: downloading {name}")
                _download_with_retry(
                    url=_HF_RESOLVE.format(repo_id=spec.repo_id, filename=name),
                    out_path=dst,
                    headers=headers,
                    cwd=repo_root,
                    status_cb=status_cb,
                    model_name=spec.local_dir,
                    filename=name,
                    max_retries=max_retries,
                    backoff_initial_s=backoff_initial_s,
                    backoff_max_s=backoff_max_s,
                )

            if weight_files:
                workers = _parallel_worker_count(len(weight_files))
                _status(
                    status_cb,
                    f"{spec.local_dir}: downloading {len(weight_files)} weight file(s) in parallel (workers={workers})",
                )
                if workers == 1:
                    for name in weight_files:
                        dst = model_dir / name
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        _status(status_cb, f"{spec.local_dir}: downloading {name}")
                        _download_with_retry(
                            url=_HF_RESOLVE.format(repo_id=spec.repo_id, filename=name),
                            out_path=dst,
                            headers=headers,
                            cwd=repo_root,
                            status_cb=status_cb,
                            model_name=spec.local_dir,
                            filename=name,
                            max_retries=max_retries,
                            backoff_initial_s=backoff_initial_s,
                            backoff_max_s=backoff_max_s,
                        )
                else:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                        future_to_name: dict[concurrent.futures.Future[None], str] = {}
                        for name in weight_files:
                            dst = model_dir / name
                            dst.parent.mkdir(parents=True, exist_ok=True)
                            _status(status_cb, f"{spec.local_dir}: queue {name}")
                            future = executor.submit(
                                _download_with_retry,
                                url=_HF_RESOLVE.format(repo_id=spec.repo_id, filename=name),
                                out_path=dst,
                                headers=headers,
                                cwd=repo_root,
                                status_cb=status_cb,
                                model_name=spec.local_dir,
                                filename=name,
                                max_retries=max_retries,
                                backoff_initial_s=backoff_initial_s,
                                backoff_max_s=backoff_max_s,
                            )
                            future_to_name[future] = name
                        for future in concurrent.futures.as_completed(future_to_name):
                            name = future_to_name[future]
                            future.result()
                            _status(status_cb, f"{spec.local_dir}: finished {name}")

            if not _is_complete_model_dir(model_dir, require_tokenizer=spec.require_tokenizer):
                raise RuntimeError(
                    f"Model download incomplete for {spec.local_dir} ({spec.repo_id}) at {model_dir}"
                )

            _normalize_config_rope_numeric_fields(model_dir)
            _status(status_cb, f"{spec.local_dir}: download complete")
            return model_dir
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def ensure_gpt2_weights_alias(repo_root: Path) -> Path:
    weights = repo_root / "models" / "gpt2" / "model.safetensors"
    if not weights.exists():
        raise RuntimeError(f"Missing GPT-2 model.safetensors at {weights}")
    return weights


def ensure_matrix_models(
    repo_root: Path,
    *,
    status_cb: Callable[[str], None] | None = None,
    model_dirs: list[str] | tuple[str, ...] | None = None,
) -> None:
    required_dirs = sorted(model_dirs) if model_dirs is not None else sorted(all_matrix_model_dirs())
    for model_dir in required_dirs:
        spec = MODEL_SPECS.get(model_dir)
        if spec is None:
            raise RuntimeError(f"No download spec registered for matrix model dir: {model_dir}")
        ensure_model_downloaded(repo_root=repo_root, spec=spec, status_cb=status_cb)
    ensure_gpt2_weights_alias(repo_root)
