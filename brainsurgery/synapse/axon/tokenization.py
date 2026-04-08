from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer


def _looks_like_hf_repo_id(value: str | None) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip()
    return "/" in normalized and " " not in normalized


def _needs_mistral_tokenizer_type(value: str | None) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip()
    return normalized in {
        "mistralai/Devstral-Small-2507",
        "mistralai/Magistral-Small-2509",
    }


def _from_pretrained(source: str, **kwargs: Any) -> Any:
    if _needs_mistral_tokenizer_type(source):
        try:
            return AutoTokenizer.from_pretrained(source, tokenizer_type="mistral", **kwargs)
        except Exception:
            pass
    return AutoTokenizer.from_pretrained(source, **kwargs)


def load_tokenizer(
    tokenizer_source: str,
    *,
    fallback_repo_id: str | None = None,
    trust_remote_code: bool = False,
) -> Any:
    candidate = Path(tokenizer_source).expanduser()
    if candidate.exists():
        source = str(candidate.resolve())
        local_attempts = [
            {"local_files_only": True, "trust_remote_code": trust_remote_code},
            {
                "local_files_only": True,
                "use_fast": False,
                "trust_remote_code": trust_remote_code,
            },
        ]
        if trust_remote_code:
            local_attempts.extend(
                [
                    {"local_files_only": True, "trust_remote_code": False},
                    {"local_files_only": True, "use_fast": False, "trust_remote_code": False},
                ]
            )
        last_error: Exception | None = None
        for kwargs in local_attempts:
            try:
                return _from_pretrained(source, **kwargs)
            except Exception as exc:
                last_error = exc
        if _looks_like_hf_repo_id(fallback_repo_id) and fallback_repo_id != source:
            try:
                return _from_pretrained(
                    fallback_repo_id,
                    local_files_only=False,
                    trust_remote_code=trust_remote_code,
                )
            except Exception:
                return _from_pretrained(
                    fallback_repo_id,
                    local_files_only=False,
                    use_fast=False,
                    trust_remote_code=trust_remote_code,
                )
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Unable to load tokenizer from local path {source}")

    try:
        return _from_pretrained(
            tokenizer_source, local_files_only=False, trust_remote_code=trust_remote_code
        )
    except Exception:
        if _looks_like_hf_repo_id(fallback_repo_id) and fallback_repo_id != tokenizer_source:
            try:
                return _from_pretrained(
                    fallback_repo_id,
                    local_files_only=False,
                    trust_remote_code=trust_remote_code,
                )
            except Exception:
                pass
        return _from_pretrained(
            tokenizer_source,
            local_files_only=False,
            use_fast=False,
            trust_remote_code=trust_remote_code,
        )


def looks_like_tokenizer_dir(path: Path) -> bool:
    if (
        (path / "tokenizer.json").exists()
        or (path / "tokenizer.model").exists()
        or ((path / "vocab.json").exists() and (path / "merges.txt").exists())
    ):
        return True
    # Some custom-code tokenizers (for example Phi-3-small) rely on
    # tokenization_*.py plus a companion tokenizer config / .tiktoken asset.
    return (path / "tokenizer_config.json").exists() and (
        any(path.glob("tokenization*.py")) or any(path.glob("*.tiktoken"))
    )


def candidate_tokenizer_dirs(model_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    candidates.append(model_dir)
    candidates.append(model_dir.with_name(f"{model_dir.name}.old"))
    parts = model_dir.name.split("_")
    for cut in range(len(parts) - 1, 1, -1):
        candidates.append(model_dir.with_name("_".join(parts[:cut])))

    out: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def _special_token_ids_from_config(
    tokenizer_source: str,
    fallback_repo_id: str | None,
    *,
    trust_remote_code: bool,
) -> tuple[int | None, int | None]:
    sources: list[str] = []
    candidate = Path(tokenizer_source).expanduser()
    if candidate.exists():
        sources.append(str(candidate.resolve()))
    else:
        sources.append(tokenizer_source)
    if _looks_like_hf_repo_id(fallback_repo_id) and fallback_repo_id not in sources:
        sources.append(fallback_repo_id)
    for source in sources:
        path = Path(source)
        try:
            if path.exists():
                cfg_path = path / "config.json"
                if cfg_path.exists():
                    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
                else:
                    continue
            else:
                cfg = AutoConfig.from_pretrained(source, trust_remote_code=trust_remote_code)
                payload = cfg.to_dict()
        except Exception:
            continue
        eos = payload.get("eos_token_id")
        bos = payload.get("bos_token_id")
        eos_id = int(eos) if isinstance(eos, int) else None
        bos_id = int(bos) if isinstance(bos, int) else None
        if eos_id is not None or bos_id is not None:
            return eos_id, bos_id
    return None, None


def _ensure_special_tokens_from_config(
    tokenizer_obj: Any,
    tokenizer_source: str,
    fallback_repo_id: str | None,
    *,
    trust_remote_code: bool,
) -> None:
    if tokenizer_obj.eos_token_id is not None or tokenizer_obj.bos_token_id is not None:
        return
    eos_id, bos_id = _special_token_ids_from_config(
        tokenizer_source,
        fallback_repo_id,
        trust_remote_code=trust_remote_code,
    )
    if eos_id is not None:
        eos_token = tokenizer_obj.convert_ids_to_tokens(eos_id)
        if isinstance(eos_token, str):
            tokenizer_obj.eos_token = eos_token
    if tokenizer_obj.bos_token_id is None and bos_id is not None:
        bos_token = tokenizer_obj.convert_ids_to_tokens(bos_id)
        if isinstance(bos_token, str):
            tokenizer_obj.bos_token = bos_token


def spec_padding_side(spec: dict[str, Any]) -> str | None:
    model = spec.get("model", {})
    if not isinstance(model, dict):
        return None
    meta = model.get("meta", {})
    if not isinstance(meta, dict):
        return None
    value = meta.get("padding_side")
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized not in {"left", "right"}:
        raise ValueError(f"Invalid model.meta.padding_side={value!r}; expected 'left' or 'right'.")
    return normalized


def preferred_padding_side(spec: dict[str, Any]) -> str:
    explicit = spec_padding_side(spec)
    if explicit is not None:
        return explicit
    return "left"


def tokenize_prompts(
    *,
    prompts: list[str],
    tokenizer_source: str,
    device: torch.device,
    max_len: int,
    lowered_spec: dict[str, Any] | None = None,
    tokenizer_fallback: str | None = None,
    trust_remote_code: bool = False,
) -> tuple[Any, torch.Tensor, torch.Tensor | None]:
    tokenizer_obj = load_tokenizer(
        tokenizer_source,
        fallback_repo_id=tokenizer_fallback,
        trust_remote_code=trust_remote_code,
    )
    _ensure_special_tokens_from_config(
        tokenizer_obj,
        tokenizer_source,
        tokenizer_fallback,
        trust_remote_code=trust_remote_code,
    )
    explicit_padding_side = spec_padding_side(lowered_spec) if lowered_spec is not None else None
    if len(prompts) > 1 or explicit_padding_side is not None:
        tokenizer_obj.padding_side = (
            explicit_padding_side
            if explicit_padding_side is not None
            else (preferred_padding_side(lowered_spec) if lowered_spec is not None else "left")
        )
        if tokenizer_obj.pad_token_id is None:
            if tokenizer_obj.eos_token_id is None:
                raise ValueError(
                    "Tokenizer has no pad token and no eos token; cannot batch prompts"
                )
            tokenizer_obj.pad_token = tokenizer_obj.eos_token
    inputs = tokenizer_obj(
        prompts,
        return_tensors="pt",
        padding=(len(prompts) > 1),
        truncation=True,
        max_length=int(max_len),
    ).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    return tokenizer_obj, input_ids, attention_mask


__all__ = [
    "candidate_tokenizer_dirs",
    "load_tokenizer",
    "looks_like_tokenizer_dir",
    "preferred_padding_side",
    "spec_padding_side",
    "tokenize_prompts",
]
