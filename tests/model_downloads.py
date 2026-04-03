from __future__ import annotations

import concurrent.futures
import json
import os
import random
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

_HF_API = "https://huggingface.co/api/models"
_HF_RESOLVE = "https://huggingface.co/{repo_id}/resolve/main/{filename}"
_ESSENTIAL_TEXT_FILES = {
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
}
_DEFAULT_PARALLEL_WORKERS = 4
_DEFAULT_MAX_RETRIES = 20
_DEFAULT_BACKOFF_INITIAL_S = 2.0
_DEFAULT_BACKOFF_MAX_S = 60.0


@dataclass(frozen=True)
class ModelDownloadSpec:
    local_dir: str
    repo_id: str
    require_tokenizer: bool = True


MODEL_SPECS: dict[str, ModelDownloadSpec] = {
    "albert": ModelDownloadSpec(local_dir="albert", repo_id="albert/albert-base-v2"),
    "apertus_8b": ModelDownloadSpec(local_dir="apertus_8b", repo_id="swiss-ai/Apertus-8B-2509"),
    "bert": ModelDownloadSpec(local_dir="bert", repo_id="google-bert/bert-base-uncased"),
    "camembert": ModelDownloadSpec(local_dir="camembert", repo_id="camembert-base"),
    "deberta_v2": ModelDownloadSpec(
        local_dir="deberta_v2",
        repo_id="microsoft/deberta-v3-xsmall",
    ),
    "distilbert": ModelDownloadSpec(
        local_dir="distilbert",
        repo_id="distilbert/distilbert-base-uncased",
    ),
    "electra": ModelDownloadSpec(
        local_dir="electra",
        repo_id="google/electra-base-generator",
    ),
    "longformer": ModelDownloadSpec(
        local_dir="longformer",
        repo_id="allenai/longformer-base-4096",
    ),
    "modernbert": ModelDownloadSpec(local_dir="modernbert", repo_id="answerdotai/ModernBERT-base"),
    "roberta": ModelDownloadSpec(local_dir="roberta", repo_id="FacebookAI/roberta-base"),
    "gpt2": ModelDownloadSpec(local_dir="gpt2", repo_id="openai-community/gpt2"),
    "t5_small": ModelDownloadSpec(local_dir="t5_small", repo_id="google-t5/t5-small"),
    "mt5_small": ModelDownloadSpec(local_dir="mt5_small", repo_id="google/mt5-small"),
    "bart_base": ModelDownloadSpec(local_dir="bart_base", repo_id="facebook/bart-base"),
    "mbart_large_50_m2m": ModelDownloadSpec(
        local_dir="mbart_large_50_m2m", repo_id="facebook/mbart-large-50-many-to-many-mmt"
    ),
    "marian_en_de": ModelDownloadSpec(
        local_dir="marian_en_de", repo_id="Helsinki-NLP/opus-mt-en-de"
    ),
    "t5gemma_s_s_ul2": ModelDownloadSpec(
        local_dir="t5gemma_s_s_ul2", repo_id="google/t5gemma-s-s-ul2"
    ),
    "t5gemma2_270m": ModelDownloadSpec(
        local_dir="t5gemma2_270m", repo_id="google/t5gemma-2-270m-270m"
    ),
    "gemma3": ModelDownloadSpec(local_dir="gemma3", repo_id="google/gemma-3-270m"),
    "gemma3_1b": ModelDownloadSpec(local_dir="gemma3_1b", repo_id="google/gemma-3-1b-pt"),
    "gemma3_4b": ModelDownloadSpec(local_dir="gemma3_4b", repo_id="google/gemma-3-4b-pt"),
    "olmoe_1b_7b_0924": ModelDownloadSpec(
        local_dir="olmoe_1b_7b_0924", repo_id="allenai/OLMoE-1B-7B-0924"
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
    "falcon_rw_1b": ModelDownloadSpec(local_dir="falcon_rw_1b", repo_id="tiiuae/falcon-rw-1b"),
    "llama3_2_1b": ModelDownloadSpec(local_dir="llama3_2_1b", repo_id="meta-llama/Llama-3.2-1B"),
    "mistral_7b_v0_1": ModelDownloadSpec(
        local_dir="mistral_7b_v0_1", repo_id="mistralai/Mistral-7B-v0.1"
    ),
    "qwen2_5_0_5b": ModelDownloadSpec(local_dir="qwen2_5_0_5b", repo_id="Qwen/Qwen2.5-0.5B"),
    "comma": ModelDownloadSpec(local_dir="comma", repo_id="common-pile/comma-v0.1-1t"),
    "dfm_decoder": ModelDownloadSpec(
        local_dir="dfm_decoder",
        repo_id="danish-foundation-models/dfm-decoder-open-v0-7b-pt",
    ),
    "smollm_135m": ModelDownloadSpec(
        local_dir="smollm_135m",
        repo_id="HuggingFaceTB/SmolLM-135M",
    ),
    "smollm_360m": ModelDownloadSpec(
        local_dir="smollm_360m",
        repo_id="HuggingFaceTB/SmolLM-360M",
    ),
    "smollm_1_7b": ModelDownloadSpec(
        local_dir="smollm_1_7b",
        repo_id="HuggingFaceTB/SmolLM-1.7B",
    ),
    "smollm2_135m": ModelDownloadSpec(
        local_dir="smollm2_135m",
        repo_id="HuggingFaceTB/SmolLM2-135M",
    ),
    "smollm2_360m": ModelDownloadSpec(
        local_dir="smollm2_360m",
        repo_id="HuggingFaceTB/SmolLM2-360M",
    ),
    "smollm2_1_7b": ModelDownloadSpec(
        local_dir="smollm2_1_7b",
        repo_id="HuggingFaceTB/SmolLM2-1.7B",
    ),
    "smollm3_3b": ModelDownloadSpec(
        local_dir="smollm3_3b",
        repo_id="HuggingFaceTB/SmolLM3-3B",
    ),
    "smollm3_3b_base": ModelDownloadSpec(
        local_dir="smollm3_3b_base",
        repo_id="HuggingFaceTB/SmolLM3-3B-Base",
    ),
    "xlm_roberta": ModelDownloadSpec(
        local_dir="xlm_roberta",
        repo_id="FacebookAI/xlm-roberta-base",
    ),
    "phi3_mini_4k_instruct": ModelDownloadSpec(
        local_dir="phi3_mini_4k_instruct",
        repo_id="microsoft/Phi-3-mini-4k-instruct",
    ),
    "mamba_tiny_random": ModelDownloadSpec(
        local_dir="mamba_tiny_random", repo_id="yujiepan/mamba-tiny-random"
    ),
    "mamba_2_8b_hf": ModelDownloadSpec(
        local_dir="mamba_2_8b_hf", repo_id="state-spaces/mamba-2.8b-hf"
    ),
    "jamba_tiny_random": ModelDownloadSpec(
        local_dir="jamba_tiny_random", repo_id="ai21labs/Jamba-tiny-random"
    ),
    "jamba_3b": ModelDownloadSpec(local_dir="jamba_3b", repo_id="ai21labs/AI21-Jamba-Reasoning-3B"),
    "glm_4_5_air": ModelDownloadSpec(local_dir="glm_4_5_air", repo_id="zai-org/GLM-4.5-Air"),
    "deepseek_v2_lite": ModelDownloadSpec(
        local_dir="deepseek_v2_lite",
        repo_id="deepseek-ai/DeepSeek-V2-Lite",
    ),
    "black_mamba_2_8b": ModelDownloadSpec(
        local_dir="black_mamba_2_8b",
        repo_id="Zyphra/BlackMamba-2.8B",
        require_tokenizer=False,
    ),
    "black_mamba": ModelDownloadSpec(
        local_dir="black_mamba_2_8b",
        repo_id="Zyphra/BlackMamba-2.8B",
        require_tokenizer=False,
    ),
    "nemotron3": ModelDownloadSpec(
        local_dir="nemotron3",
        repo_id="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    ),
    "flexmath": ModelDownloadSpec(local_dir="flexmath", repo_id="allenai/Flex-math-2x7B-1T"),
}

MATRIX_AXON_TO_MODEL_DIR: dict[str, str] = {
    "albert": "albert",
    "apertus_8b": "apertus_8b",
    "bert": "bert",
    "deberta_v2": "deberta_v2",
    "distilbert": "distilbert",
    "electra": "electra",
    "longformer": "longformer",
    "modernbert": "modernbert",
    "falcon_rw_1b": "falcon_rw_1b",
    "flexolmo": "flexmath",
    "gemma3_270m": "gemma3",
    "gpt2": "gpt2",
    "gpt2_kv": "gpt2",
    "jamba_3b": "jamba_3b",
    "black_mamba": "black_mamba_2_8b",
    "dfm_decoder": "dfm_decoder",
    "llama3_2_1b": "llama3_2_1b",
    "mamba_2_8b": "mamba_2_8b_hf",
    "mistral_7b_v0_1": "mistral_7b_v0_1",
    "olmo3": "olmo3_1025_7b",
    "olmoe_1b_7b_0924": "olmoe_1b_7b_0924",
    "olmo_2_1b": "olmo_2_1b",
    "qwen2_5_0_5b": "qwen2_5_0_5b",
    "roberta": "roberta",
    "smollm": "smollm_135m",
    "smollm3": "smollm3_3b_base",
    "t5_small": "t5_small",
    "mt5": "mt5_small",
    "bart": "bart_base",
    "mbart": "mbart_large_50_m2m",
    "marian": "marian_en_de",
    "t5gemma": "t5gemma_s_s_ul2",
    "t5gemma2": "t5gemma2_270m",
}

# Matrix pairs allow multiple model dirs to share one Axon file stem.
MATRIX_AXON_MODEL_DIR_PAIRS: list[tuple[str, str]] = [
    *sorted(MATRIX_AXON_TO_MODEL_DIR.items()),
    ("roberta", "camembert"),
    ("roberta", "xlm_roberta"),
    ("dfm_decoder", "comma"),
]


def _status(config: pytest.Config, message: str) -> None:
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(f"[model-download] {message}")
    else:
        print(f"[model-download] {message}", flush=True)


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
        for h in headers:
            cmd.extend(["-H", h])
    cmd.extend(["-o", str(out_path), url])
    run = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if run.returncode != 0:
        raise RuntimeError(f"curl failed for {url}\nstdout:\n{run.stdout}\nstderr:\n{run.stderr}")


def _download_with_retry(
    *,
    url: str,
    out_path: Path,
    headers: list[str],
    cwd: Path,
    config: pytest.Config,
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
                config,
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
        # If safetensors is unavailable, fall back to existence checks.
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


def _load_hf_siblings(*, repo_id: str, cwd: Path) -> list[str]:
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
    out: list[str] = []
    for item in siblings:
        if isinstance(item, dict) and isinstance(item.get("rfilename"), str):
            out.append(item["rfilename"])
    return out


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
    if require_tokenizer and not (model_dir / "tokenizer_config.json").exists():
        return False
    if require_tokenizer:
        has_tokenizer = (
            (model_dir / "tokenizer.json").exists()
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


def ensure_model_downloaded(
    *,
    repo_root: Path,
    config: pytest.Config,
    spec: ModelDownloadSpec,
) -> Path:
    if shutil.which("curl") is None:
        raise RuntimeError("curl is required to download test models")

    model_dir = repo_root / "models" / spec.local_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    _normalize_local_weight_format(model_dir)

    if _is_complete_model_dir(model_dir, require_tokenizer=spec.require_tokenizer):
        _normalize_config_rope_numeric_fields(model_dir)
        _status(config, f"{spec.local_dir}: already complete, skipping download")
        return model_dir

    _status(config, f"{spec.local_dir}: fetching sibling manifest from {spec.repo_id}")
    siblings = _load_hf_siblings(repo_id=spec.repo_id, cwd=repo_root)

    shard_files = sorted(name for name in siblings if name.endswith(".safetensors"))
    pytorch_bin_files = sorted(name for name in siblings if name.endswith(".bin"))
    index_files = [name for name in siblings if name == "model.safetensors.index.json"]
    if shard_files:
        selected_weight_files = [*index_files, *shard_files]
        # Prefer safetensors whenever available to avoid ambiguous mixed-format directories.
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
        _status(config, f"{spec.local_dir}: download complete")
        return model_dir

    text_files = [name for name in pending_files if name in _ESSENTIAL_TEXT_FILES]
    weight_files = [name for name in pending_files if name not in _ESSENTIAL_TEXT_FILES]

    for name in text_files:
        dst = model_dir / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        _status(config, f"{spec.local_dir}: downloading {name}")
        _download_with_retry(
            url=_HF_RESOLVE.format(repo_id=spec.repo_id, filename=name),
            out_path=dst,
            headers=headers,
            cwd=repo_root,
            config=config,
            model_name=spec.local_dir,
            filename=name,
            max_retries=int(os.environ.get("MODEL_DOWNLOAD_MAX_RETRIES", _DEFAULT_MAX_RETRIES)),
            backoff_initial_s=float(
                os.environ.get("MODEL_DOWNLOAD_BACKOFF_INITIAL_S", _DEFAULT_BACKOFF_INITIAL_S)
            ),
            backoff_max_s=float(
                os.environ.get("MODEL_DOWNLOAD_BACKOFF_MAX_S", _DEFAULT_BACKOFF_MAX_S)
            ),
        )

    if weight_files:
        workers = _parallel_worker_count(len(weight_files))
        _status(
            config,
            f"{spec.local_dir}: downloading {len(weight_files)} weight file(s) in parallel (workers={workers})",
        )
        max_retries = int(os.environ.get("MODEL_DOWNLOAD_MAX_RETRIES", _DEFAULT_MAX_RETRIES))
        backoff_initial_s = float(
            os.environ.get("MODEL_DOWNLOAD_BACKOFF_INITIAL_S", _DEFAULT_BACKOFF_INITIAL_S)
        )
        backoff_max_s = float(
            os.environ.get("MODEL_DOWNLOAD_BACKOFF_MAX_S", _DEFAULT_BACKOFF_MAX_S)
        )
        if workers == 1:
            for name in weight_files:
                dst = model_dir / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                _status(config, f"{spec.local_dir}: downloading {name}")
                _download_with_retry(
                    url=_HF_RESOLVE.format(repo_id=spec.repo_id, filename=name),
                    out_path=dst,
                    headers=headers,
                    cwd=repo_root,
                    config=config,
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
                    _status(config, f"{spec.local_dir}: queue {name}")
                    future = executor.submit(
                        _download_with_retry,
                        url=_HF_RESOLVE.format(repo_id=spec.repo_id, filename=name),
                        out_path=dst,
                        headers=headers,
                        cwd=repo_root,
                        config=config,
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
                    _status(config, f"{spec.local_dir}: finished {name}")

    if not _is_complete_model_dir(model_dir, require_tokenizer=spec.require_tokenizer):
        raise RuntimeError(
            f"Model download incomplete for {spec.local_dir} ({spec.repo_id}) at {model_dir}"
        )

    _normalize_config_rope_numeric_fields(model_dir)
    _status(config, f"{spec.local_dir}: download complete")
    return model_dir


def ensure_gpt2_weights_alias(repo_root: Path, config: pytest.Config) -> Path:
    # Legacy helper name kept for call sites; we now consume GPT-2 directly from models/gpt2.
    weights = repo_root / "models" / "gpt2" / "model.safetensors"
    if not weights.exists():
        raise RuntimeError(f"Missing GPT-2 model.safetensors at {weights}")
    return weights


def ensure_matrix_models(repo_root: Path, config: pytest.Config) -> None:
    required_dirs = sorted({model_dir for _, model_dir in MATRIX_AXON_MODEL_DIR_PAIRS})
    for model_dir in required_dirs:
        spec = MODEL_SPECS.get(model_dir)
        if spec is None:
            raise RuntimeError(f"No download spec registered for matrix model dir: {model_dir}")
        ensure_model_downloaded(repo_root=repo_root, config=config, spec=spec)

    # Keep GPT-2 single-file path validation in one place for test setup.
    ensure_gpt2_weights_alias(repo_root, config)
