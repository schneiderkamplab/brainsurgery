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
    "gpt2.old": ModelDownloadSpec(local_dir="gpt2.old", repo_id="openai-community/gpt2"),
    "gemma3": ModelDownloadSpec(local_dir="gemma3", repo_id="google/gemma-3-270m"),
    "gemma3_1b": ModelDownloadSpec(local_dir="gemma3_1b", repo_id="google/gemma-3-1b-pt"),
    "gemma3_4b": ModelDownloadSpec(local_dir="gemma3_4b", repo_id="google/gemma-3-4b-pt"),
    "olmoe_1b_7b_0924": ModelDownloadSpec(
        local_dir="olmoe_1b_7b_0924", repo_id="allenai/OLMoE-1B-7B-0924"
    ),
    "falcon_rw_1b": ModelDownloadSpec(local_dir="falcon_rw_1b", repo_id="tiiuae/falcon-rw-1b"),
    "llama3_2_1b": ModelDownloadSpec(local_dir="llama3_2_1b", repo_id="meta-llama/Llama-3.2-1B"),
    "mistral_7b_v0_1": ModelDownloadSpec(
        local_dir="mistral_7b_v0_1", repo_id="mistralai/Mistral-7B-v0.1"
    ),
    "qwen2_5_0_5b": ModelDownloadSpec(local_dir="qwen2_5_0_5b", repo_id="Qwen/Qwen2.5-0.5B"),
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
    "falcon_rw_1b": "falcon_rw_1b",
    "flexolmo": "flexmath",
    "gemma3_270m": "gemma3",
    "gpt2": "gpt2.old",
    "gpt2_kv": "gpt2.old",
    "jamba_3b": "jamba_3b",
    "black_mamba": "black_mamba_2_8b",
    "llama3_2_1b": "llama3_2_1b",
    "mamba_2_8b": "mamba_2_8b_hf",
    "mistral_7b_v0_1": "mistral_7b_v0_1",
    "olmoe_1b_7b_0924": "olmoe_1b_7b_0924",
    "qwen2_5_0_5b": "qwen2_5_0_5b",
}


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
        has_tokenizer = (model_dir / "tokenizer.json").exists() or (
            (model_dir / "vocab.json").exists() and (model_dir / "merges.txt").exists()
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

    if changed:
        config_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


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

    if _is_complete_model_dir(model_dir, require_tokenizer=spec.require_tokenizer):
        _normalize_config_rope_numeric_fields(model_dir)
        _status(config, f"{spec.local_dir}: already complete, skipping download")
        return model_dir

    _status(config, f"{spec.local_dir}: fetching sibling manifest from {spec.repo_id}")
    siblings = _load_hf_siblings(repo_id=spec.repo_id, cwd=repo_root)

    shard_files = sorted(name for name in siblings if name.endswith(".safetensors"))
    pytorch_bin_files = sorted(name for name in siblings if name.endswith(".bin"))
    index_files = [name for name in siblings if name == "model.safetensors.index.json"]
    selected_files = set(index_files + shard_files + pytorch_bin_files)
    selected_files.update(name for name in siblings if name in _ESSENTIAL_TEXT_FILES)

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
    src = repo_root / "models" / "gpt2.old" / "model.safetensors"
    if not src.exists():
        raise RuntimeError(f"Missing source GPT-2 model.safetensors at {src}")

    dst = repo_root / "models" / "gpt2" / "model.safetensors"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        _status(config, "gpt2/model.safetensors already present, skipping")
        return dst

    _status(config, "creating gpt2/model.safetensors alias from gpt2.old/model.safetensors")
    shutil.copy2(src, dst)
    return dst


def ensure_matrix_models(repo_root: Path, config: pytest.Config) -> None:
    required_dirs = sorted(set(MATRIX_AXON_TO_MODEL_DIR.values()))
    for model_dir in required_dirs:
        spec = MODEL_SPECS.get(model_dir)
        if spec is None:
            raise RuntimeError(f"No download spec registered for matrix model dir: {model_dir}")
        ensure_model_downloaded(repo_root=repo_root, config=config, spec=spec)

    # Keep legacy GPT-2 single-file path working for tests that use models/gpt2/model.safetensors.
    ensure_gpt2_weights_alias(repo_root, config)
